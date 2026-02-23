

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List
from transformers import AutoModelForCausalLM

from .paper_compliant_router import (
    PaperCompliantPhaseRouter,
    compute_router_policy_gradient_loss,
)
from .paper_compliant_experts import (
    PaperCompliantExpertMixture,
    compute_diversity_loss,
    compute_balance_loss,
)

class PaperCompliantPAMoEActor(nn.Module):

    def __init__(
        self,
        base_model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        num_experts: int = 4,
        lora_rank: int = 32,
        device: str = "cuda",
        tau_0: float = 2.0,
        tau_f: float = 0.5,
        anneal_steps: int = 3000,
        lambda_s: float = 0.05,
        tau_div: float = 0.1,
        lambda_div: float = 0.01,
        lambda_bal: float = 0.001,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.device = device
        
        self.tau_div = tau_div
        self.lambda_div = lambda_div
        self.lambda_bal = lambda_bal
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device if torch.cuda.is_available() else "cpu",
        )
        hidden_dim = base_model.config.hidden_size
        
        self.router = PaperCompliantPhaseRouter(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            history_window=5,
            tau_0=tau_0,
            tau_f=tau_f,
            anneal_steps=anneal_steps,
            lambda_s=lambda_s,
        )
        
        self.experts = PaperCompliantExpertMixture(
            base_model=base_model,
            num_experts=num_experts,
            lora_rank=lora_rank,
        )
        
        self.reset_trajectory_buffer()
    
    def reset_trajectory_buffer(self):
        
        self.trajectory_buffer = {
            'log_probs_router': [],   # For router policy gradient
            'phase_modes': [],         # For switching penalty
            'phase_probs': [],         # For balance loss
            'expert_outputs': [],      # For diversity loss (optional)
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        obs_hidden: Optional[torch.Tensor] = None,
        goal_hidden: Optional[torch.Tensor] = None,
        history_hidden: Optional[torch.Tensor] = None,
        return_expert_info: bool = False,
        **kwargs,
    ):
        
        batch_size = input_ids.shape[0]
        
        if obs_hidden is not None and goal_hidden is not None and history_hidden is not None:
            phase_probs, phase_logits, phase_modes = self.router(
                obs_hidden=obs_hidden,
                goal_hidden=goal_hidden,
                history_hidden=history_hidden,
            )
            
            expert_ids, log_probs_router = self.router.select_expert(
                phase_probs,
                deterministic=not self.training,
            )
        else:
            print("WARNING: Router inputs not provided, using random expert selection")
            expert_ids = torch.randint(0, self.num_experts, (batch_size,), device=self.device)
            phase_probs = torch.ones(batch_size, self.num_experts, device=self.device) / self.num_experts
            phase_modes = expert_ids
            log_probs_router = torch.zeros(batch_size, device=self.device)
        
        if self.training:
            self.trajectory_buffer['log_probs_router'].append(log_probs_router)
            self.trajectory_buffer['phase_modes'].append(phase_modes)
            self.trajectory_buffer['phase_probs'].append(phase_probs)
        
        output = self.experts.forward_with_routing(
            expert_ids=expert_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        
        if return_expert_info:
            expert_info = {
                'expert_ids': expert_ids,
                'phase_probs': phase_probs,
                'phase_modes': phase_modes,
                'log_probs_router': log_probs_router,
                'temperature': self.router.current_temperature.item(),
            }
            return output, expert_info
        else:
            return output
    
    def compute_pa_moe_losses(
        self,
        trajectory_return: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        
        losses = {}
        
        if len(self.trajectory_buffer['log_probs_router']) > 0:
            log_probs_router = torch.stack(self.trajectory_buffer['log_probs_router'])  # (T, batch)
            losses['router_pg'] = compute_router_policy_gradient_loss(
                log_probs_router.mean(dim=1),
                trajectory_return,
            )
        
        if len(self.trajectory_buffer['phase_modes']) > 0:
            phase_modes_seq = torch.stack(self.trajectory_buffer['phase_modes'])  # (T, batch)
            losses['switching'] = self.router.compute_switching_penalty(phase_modes_seq)
        
        if len(self.trajectory_buffer['phase_probs']) > 0:
            avg_phase_probs = torch.stack(self.trajectory_buffer['phase_probs']).mean(dim=0)  # (batch, K)
            losses['balance'] = compute_balance_loss(
                avg_phase_probs,
                lambda_bal=self.lambda_bal,
            )

        return losses
    
    def step(self):
        
        new_temp = self.router.update_temperature()
        return new_temp
    
    def get_statistics(self) -> dict:
        
        stats = self.router.get_statistics()
        
        if len(self.trajectory_buffer['expert_ids']) > 0:
            expert_ids = torch.cat(self.trajectory_buffer['expert_ids'])
            stats['trajectory/avg_expert_id'] = expert_ids.float().mean().item()
            stats['trajectory/length'] = len(self.trajectory_buffer['expert_ids'])
        
        return stats

def create_paper_compliant_pa_moe(
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_experts: int = 4,
    lora_rank: int = 32,
    device: str = "cuda",
    **kwargs,
) -> PaperCompliantPAMoEActor:
    
    return PaperCompliantPAMoEActor(
        base_model_path=model_path,
        num_experts=num_experts,
        lora_rank=lora_rank,
        device=device,
        tau_0=2.0,
        tau_f=0.5,
        anneal_steps=3000,
        lambda_s=0.05,
        tau_div=0.1,
        lambda_div=0.01,
        lambda_bal=0.001,
        **kwargs,
    )

if __name__ == "__main__":
    actor = create_paper_compliant_pa_moe(
        model_path="Qwen/Qwen2.5-1.5B-Instruct",
        num_experts=4,
        lora_rank=32,
        device="cpu",
    )
    
    actor.train()
    actor.reset_trajectory_buffer()
    
    T = 10
    batch_size = 2
    
    for t in range(T):
        input_ids = torch.randint(0, 1000, (batch_size, 20))
        obs_hidden = torch.randn(batch_size, 15, 1536)
        goal_hidden = torch.randn(batch_size, 10, 1536)
        history_hidden = torch.randn(batch_size, 5, 1536)
        
        output, expert_info = actor(
            input_ids=input_ids,
            obs_hidden=obs_hidden,
            goal_hidden=goal_hidden,
            history_hidden=history_hidden,
            return_expert_info=True,
        )
    
    trajectory_return = torch.tensor(5.0)
    losses = actor.compute_pa_moe_losses(trajectory_return)
    
    for step in range(5):
        new_temp = actor.step()
