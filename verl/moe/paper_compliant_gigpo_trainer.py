

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np

from verl.moe.paper_compliant_pa_moe_actor import PaperCompliantPAMoEActor
from gigpo.core_gigpo import compute_gigpo_outcome_advantage

class PAMoEGiGPOTrainer:

    def __init__(
        self,
        pa_moe_actor: PaperCompliantPAMoEActor,
        actor_lr: float = 5e-6,
        router_lr: float = 1e-4,
        clip_eps: float = 0.2,
        max_grad_norm: float = 1.0,
    ):
        self.pa_moe_actor = pa_moe_actor
        self.clip_eps = clip_eps
        self.max_grad_norm = max_grad_norm
        
        self.router_optimizer = torch.optim.AdamW(
            pa_moe_actor.router.parameters(),
            lr=router_lr,
        )
        
        self.expert_optimizer = torch.optim.AdamW(
            pa_moe_actor.experts.parameters(),
            lr=actor_lr,
        )
        
        print(f"\n{'='*60}")
        print(f"PA-MoE + GiGPO Trainer Initialized")
        print(f"  Router LR: {router_lr}")
        print(f"  Expert LR: {actor_lr}")
        print(f"  Clip ε: {clip_eps}")
        print(f"{'='*60}\n")
    
    def compute_gigpo_advantages_with_routing(
        self,
        token_level_rewards: torch.Tensor,
        step_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        anchor_obs: np.ndarray,
        index: np.ndarray,
        traj_index: np.ndarray,
        step_advantage_w: float = 1.0,
        enable_similarity: bool = False,
        similarity_thresh: float = 0.95,
    ):
        
        advantages, returns = compute_gigpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            step_rewards=step_rewards,
            response_mask=response_mask,
            anchor_obs=anchor_obs,
            index=index,
            traj_index=traj_index,
            step_advantage_w=step_advantage_w,
            mode="mean_norm",
            enable_similarity=enable_similarity,
            similarity_thresh=similarity_thresh,
        )
        
        return advantages, returns
    
    def compute_policy_loss(
        self,
        log_probs_old: torch.Tensor,
        log_probs_new: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        ratio = torch.exp(log_probs_new - log_probs_old)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        
        policy_loss = -torch.min(surr1, surr2)
        
        policy_loss = (policy_loss * response_mask).sum() / response_mask.sum()
        
        return policy_loss
    
    def train_step(
        self,
        batch: Dict,
        trajectory_return: torch.Tensor,
    ) -> Dict[str, float]:
        
        metrics = {}
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        obs_hidden = batch['obs_hidden']
        goal_hidden = batch['goal_hidden']
        history_hidden = batch['history_hidden']
        
        token_level_rewards = batch['token_level_rewards']
        step_rewards = batch['step_rewards']
        response_mask = batch['response_mask']
        anchor_obs = batch['anchor_obs']
        index = batch['index']
        traj_index = batch['traj_index']
        
        log_probs_old = batch['log_probs_old']  # From rollout
        
        advantages, returns = self.compute_gigpo_advantages_with_routing(
            token_level_rewards=token_level_rewards,
            step_rewards=step_rewards,
            response_mask=response_mask,
            anchor_obs=anchor_obs,
            index=index,
            traj_index=traj_index,
        )
        
        metrics['advantage_mean'] = advantages.mean().item()
        metrics['advantage_std'] = advantages.std().item()
        
        self.pa_moe_actor.reset_trajectory_buffer()
        
        output, expert_info = self.pa_moe_actor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            obs_hidden=obs_hidden,
            goal_hidden=goal_hidden,
            history_hidden=history_hidden,
            return_expert_info=True,
        )
        
        logits = output.logits
        log_probs_new = torch.log_softmax(logits, dim=-1)
        
        metrics['temperature'] = expert_info['temperature']
        metrics['expert_entropy'] = -(expert_info['phase_probs'] * 
                                      torch.log(expert_info['phase_probs'] + 1e-10)).sum(dim=-1).mean().item()

        policy_loss = self.compute_policy_loss(
            log_probs_old=log_probs_old,
            log_probs_new=log_probs_new,
            advantages=advantages,
            response_mask=response_mask,
        )
        metrics['loss/policy'] = policy_loss.item()
        
        pa_moe_losses = self.pa_moe_actor.compute_pa_moe_losses(trajectory_return)
        
        for name, loss in pa_moe_losses.items():
            metrics[f'loss/{name}'] = loss.item()
        
        total_loss = policy_loss
        
        if 'router_pg' in pa_moe_losses:
            total_loss += pa_moe_losses['router_pg']
        
        if 'switching' in pa_moe_losses:
            total_loss += pa_moe_losses['switching']
        
        if 'balance' in pa_moe_losses:
            total_loss += pa_moe_losses['balance']
        
        if 'diversity' in pa_moe_losses:
            total_loss += pa_moe_losses['diversity']
        
        metrics['loss/total'] = total_loss.item()

        self.router_optimizer.zero_grad()
        self.expert_optimizer.zero_grad()
        
        total_loss.backward()
        
        router_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.pa_moe_actor.router.parameters(),
            self.max_grad_norm,
        )
        expert_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.pa_moe_actor.experts.parameters(),
            self.max_grad_norm,
        )
        
        metrics['grad_norm/router'] = router_grad_norm.item()
        metrics['grad_norm/expert'] = expert_grad_norm.item()
        
        self.router_optimizer.step()
        self.expert_optimizer.step()
        
        new_temp = self.pa_moe_actor.step()
        metrics['temperature_updated'] = new_temp
        
        return metrics
    
    def train_epoch(
        self,
        dataloader,
        epoch: int,
    ) -> Dict[str, float]:
        
        self.pa_moe_actor.train()
        
        epoch_metrics = {}
        all_metrics = []
        
        for batch_idx, batch in enumerate(dataloader):
            trajectory_return = batch.get('trajectory_return', torch.tensor(0.0))
            
            metrics = self.train_step(batch, trajectory_return)
            all_metrics.append(metrics)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss={metrics['loss/total']:.4f}, "
                      f"Temp={metrics['temperature']:.3f}")
        
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            epoch_metrics[key] = sum(values) / len(values)
        
        return epoch_metrics

def train_pa_moe_gigpo(
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_experts: int = 4,
    lora_rank: int = 32,
    num_epochs: int = 10,
    device: str = "cuda",
):
    
    from verl.moe.paper_compliant_pa_moe_actor import create_paper_compliant_pa_moe
    
    print("=" * 80)
    print("Training PA-MoE + GiGPO (Paper-Compliant)")
    print("=" * 80)
    
    pa_moe_actor = create_paper_compliant_pa_moe(
        model_path=model_path,
        num_experts=num_experts,
        lora_rank=lora_rank,
        device=device,
    )
    
    trainer = PAMoEGiGPOTrainer(
        pa_moe_actor=pa_moe_actor,
        actor_lr=5e-6,
        router_lr=1e-4,
        clip_eps=0.2,
    )
    
    print("\nTraining would start here with actual dataloader...")
    print("Key points:")
    print("  - Router selects expert at each step")
    print("  - GiGPO computes advantages")
    print("  - Both router and experts are updated")
    print("  - All auxiliary losses are applied")
    print("  - Temperatals from 2.0 to 0.5")

    print("\n" + "=" * 80)
    print("Train ready!")
    print("\nTo use:")
    print("  1. Prepare your dataloader with required fields")
    print("  2. Call trainer.train_aloader, epoch)")
    print("  3. Monitor metrics including switching penalty, temperature, etc.")
    print("=" * 80)

if __name__ == "__main__":
    train_pa_moe_gigpo(
        model_path="Qwen/Qwen2.5-1.5B-Instruct",
        num_experts=4,
        lora_rank=32,
        device="cpu",
    )
