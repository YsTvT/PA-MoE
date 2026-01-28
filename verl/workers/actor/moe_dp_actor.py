

import itertools
import time
import logging
import os
from typing import Tuple, Dict, Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.moe.moe_algos import (
    compute_moe_policy_loss,
    compute_load_balancing_loss,
    select_expert_winners,
)
from verl.moe.moe_actor import MoEActor
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.actor import BasePPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class MoEDataParallelActor(BasePPOActor):

    def __init__(
        self,
        config,
        moe_actor: MoEActor,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        super().__init__(config)
        self.moe_actor = moe_actor
        self.actor_optimizer = actor_optimizer
        
        self.num_experts = moe_actor.num_experts
        self.use_winner_take_all = config.get('use_winner_take_all', True)
        self.load_balance_weight = config.get('load_balance_weight', 0.01)
        self.competition_weight = config.get('competition_weight', 0.05)
        
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        self.device_name = get_device_name()
        
        self.expert_stats = {
            'usage_count': torch.zeros(self.num_experts),
            'win_count': torch.zeros(self.num_experts),
        }
    
    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        
        response_length = micro_batch["responses"].size(-1)
        
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            
            output = self.moe_actor(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                temperature=temperature,
                training=self.moe_actor.training,
            )
            
            logits = output.logits
            gates = output.gates
            load_balancing_loss = output.load_balancing_loss
            competition_loss = output.competition_loss
            
            logits = logits[:, -response_length - 1 : -1, :]
            
            log_probs = logprobs_from_logits(
                logits=logits,
                labels=micro_batch["responses"],
                inplace_backward=not calculate_entropy,
            )
            
            entropy = None
            if calculate_entropy:
                entropy = verl_F.entropy_from_logits(logits)
            
            moe_info = {
                'gates': gates,
                'top_k_indices': output.top_k_indices,
                'load_balancing_loss': load_balancing_loss,
                'competition_loss': competition_loss,
            }
        
        return entropy, log_probs, moe_info
    
    def _optimizer_step(self):
        
        assert self.config.grad_clip is not None
        
        if isinstance(self.moe_actor, FSDP):
            grad_norm = self.moe_actor.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.moe_actor, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(
                self.moe_actor.parameters(),
                max_norm=self.config.grad_clip
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.moe_actor.parameters(),
                max_norm=self.config.grad_clip
            )
        
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        
        return grad_norm
    
    @GPUMemoryLogger(role="moe actor", logger=logger)
    def compute_log_prob(
        self,
        data: DataProto,
        calculate_entropy=False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        self.moe_actor.eval()
        
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        micro_batches = batch.split(micro_batch_size)
        
        log_probs_lst = []
        entropy_lst = []
        gates_lst = []
        
        for micro_batch in micro_batches:
            with torch.no_grad():
                entropy, log_probs, moe_info = self._forward_micro_batch(
                    micro_batch,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                )
            
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            gates_lst.append(moe_info['gates'])
        
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropies = None
        if calculate_entropy:
            entropies = torch.concat(entropy_lst, dim=0)
        gates = torch.concat(gates_lst, dim=0)
        
        with torch.no_grad():
            usage = (gates > 0).float().sum(dim=0)
            self.expert_stats['usage_count'] += usage.cpu()
        
        return log_probs, entropies
    
    @GPUMemoryLogger(role="moe actor", logger=logger)
    def update_policy(self, data: DataProto) -> Dict:
        
        self.moe_actor.train()
        
        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)
        
        select_keys = [
            "responses", "input_ids", "attention_mask", "position_ids",
            "old_log_probs", "advantages"
        ]
        if multi_turn:
            select_keys.append("loss_mask")
        
        batch = data.select(batch_keys=select_keys).batch
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        
        metrics = {}
        
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(dataloader):
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                
                self.actor_optimizer.zero_grad()
                
                total_policy_loss = 0.0
                total_load_loss = 0.0
                total_competition_loss = 0.0
                batch_moe_info = []
                
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_torch_device().current_device())
                    
                    responses = micro_batch["responses"]
                    response_length = responses.size(1)
                    attention_mask = micro_batch["attention_mask"]
                    
                    if multi_turn:
                        response_mask = micro_batch["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]
                    
                    old_log_prob = micro_batch["old_log_probs"]
                    advantages = micro_batch["advantages"]
                    
                    calculate_entropy = self.config.entropy_coeff != 0
                    entropy, log_prob, moe_info = self._forward_micro_batch(
                        micro_batch=micro_batch,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )
                    
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_moe_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        gates=moe_info['gates'],
                        expert_masks=None,
                        cliprange=self.config.clip_ratio,
                        cliprange_low=self.config.clip_ratio_low,
                        cliprange_high=self.config.clip_ratio_high,
                        loss_agg_mode=self.config.loss_agg_mode,
                        use_winner_take_all=self.use_winner_take_all,
                    )
                    
                    policy_loss = pg_loss
                    if calculate_entropy:
                        from verl.trainer.ppo.core_algos import agg_loss
                        entropy_loss = agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode=self.config.loss_agg_mode,
                        )
                        policy_loss = policy_loss - entropy_loss * self.config.entropy_coeff
                    
                    load_loss = moe_info['load_balancing_loss'] * self.load_balance_weight
                    
                    competition_loss = moe_info['competition_loss'] * self.competition_weight
                    
                    total_loss = policy_loss + load_loss + competition_loss
                    
                    gradient_accumulation = (
                        self.config.ppo_mini_batch_size // 
                        self.config.ppo_micro_batch_size_per_gpu
                    )
                    loss = total_loss / gradient_accumulation
                    loss.backward()
                    
                    total_policy_loss += policy_loss.detach().item()
                    total_load_loss += load_loss.detach().item()
                    total_competition_loss += competition_loss.detach().item()
                    batch_moe_info.append(moe_info)
                    
                    append_to_dict(metrics, {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    })
                
                grad_norm = self._optimizer_step()
                
                avg_gates = torch.cat([info['gates'] for info in batch_moe_info], dim=0).mean(dim=0)
                append_to_dict(metrics, {
                    "actor/grad_norm": grad_norm.detach().item(),
                    "moe/load_balance_loss": total_load_loss / len(micro_batches),
                    "moe/competition_loss": total_competition_loss / len(micro_batches),
                    "moe/expert_usage_variance": avg_gates.var().item(),
                })
                
                for i in range(self.num_experts):
                    append_to_dict(metrics, {
                        f"moe/expert_{i}_usage": avg_gates[i].item(),
                    })
        
        self.actor_optimizer.zero_grad()
        
        usage_stats = self.moe_actor.get_usage_stats()
        for key, value in usage_stats.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    metrics[f"moe/{key}"] = value.item()
                else:
                    for i, v in enumerate(value):
                        metrics[f"moe/{key}_expert_{i}"] = v.item() if isinstance(v, torch.Tensor) else v
        
        return metrics
    
    def get_expert_stats(self) -> Dict:
        
        return {
            'usage_count': self.expert_stats['usage_count'].clone(),
            'win_count': self.expert_stats['win_count'].clone(),
        }
    
    def reset_expert_stats(self):
        
        self.expert_stats = {
            'usage_count': torch.zeros(self.num_experts),
            'win_count': torch.zeros(self.num_experts),
        }
        self.moe_actor.reset_stats()
