

import torch
import torch.nn as nn
from typing import Optional, Dict
from transformers import AutoModelForCausalLM

from verl.moe.paper_compliant_router import (
    PaperCompliantPhaseRouter,
    compute_router_policy_gradient_loss,
)

class K0Configuration(nn.Module):

    def __init__(
        self,
        base_model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        tau_0: float = 2.0,
        tau_f: float = 0.5,
        anneal_steps: int = 3000,
        lambda_s: float = 0.05,
    ):
        super().__init__()
        
        self.device = device
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device if torch.cuda.is_available() else "cpu",
        )
        hidden_dim = self.base_model.config.hidden_size
        
        self.router = PaperCompliantPhaseRouter(
            hidden_dim=hidden_dim,
            num_experts=4,
            history_window=5,
            tau_0=tau_0,
            tau_f=tau_f,
            anneal_steps=anneal_steps,
            lambda_s=lambda_s,
        )
        
        self.reset_trajectory_buffer()
    
    def reset_trajectory_buffer(self):
        
        self.trajectory_buffer = {
            'log_probs_router': [],
            'phase_modes': [],
            'phase_probs': [],
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        obs_hidden: Optional[torch.Tensor] = None,
        goal_hidden: Optional[torch.Tensor] = None,
        history_hidden: Optional[torch.Tensor] = None,
        return_routing_info: bool = False,
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
            expert_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            phase_probs = torch.ones(batch_size, 4, device=self.device) / 4
            phase_modes = expert_ids
            log_probs_router = torch.zeros(batch_size, device=self.device)
        
        if self.training:
            self.trajectory_buffer['log_probs_router'].append(log_probs_router)
            self.trajectory_buffer['phase_modes'].append(phase_modes)
            self.trajectory_buffer['phase_probs'].append(phase_probs)
        
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        
        if return_routing_info:
            routing_info = {
                'expert_ids': expert_ids,        # Predicted but not used
                'phase_probs': phase_probs,
                'phase_modes': phase_modes,
                'log_probs_router': log_probs_router,
                'temperature': self.router.current_temperature.item(),
                'note': 'expert_ids are predicted but all use base model (K=0)',
            }
            return output, routing_info
        else:
            return output
    
    def compute_routing_losses(
        self,
        trajectory_return: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        
        losses = {}
        
        if len(self.trajectory_buffer['log_probs_router']) > 0:
            log_probs_router = torch.stack(self.trajectory_buffer['log_probs_router'])
            losses['router_pg'] = compute_router_policy_gradient_loss(
                log_probs_router.mean(dim=1),
                trajectory_return,
            )
        
        if len(self.trajectory_buffer['phase_modes']) > 0:
            phase_modes_seq = torch.stack(self.trajectory_buffer['phase_modes'])
            losses['switching'] = self.router.compute_switching_penalty(phase_modes_seq)
        
        return losses
    
    def step(self):
        
        return self.router.update_temperature()
    
    def get_statistics(self) -> dict:
        
        stats = self.router.get_statistics()
        stats['config'] = 'K=0 (router-only)'
        return stats

def create_k0_configuration(
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda",
) -> K0Configuration:
    
    return K0Configuration(
        base_model_path=model_path,
        device=device,
        tau_0=2.0,
        tau_f=0.5,
        anneal_steps=3000,
        lambda_s=0.05,
    )

if __name__ == "__main__":
    k0_model = create_k0_configuration(
        model_path="Qwen/Qwen2.5-1.5B-Instruct",
        device="cpu",
    )
    
    k0_model.train()
    k0_model.reset_trajectory_buffer()
    
    batch_size = 2
    input_ids = torch.randint(0, 1000, (batch_size, 20))
    obs_hidden = torch.randn(batch_size, 15, 1536)
    goal_hidden = torch.randn(batch_size, 10, 1536)
    history_hidden = torch.randn(batch_size, 5, 1536)
    
    output, routing_info = k0_model(
        input_ids=input_ids,
        obs_hidden=obs_hidden,
        goal_hidden=goal_hidden,
        history_hidden=history_hidden,
        return_routing_info=True,
    )
    
    trajectory_return = torch.tensor(5.0)
    losses = k0_model.compute_routing_losses(trajectory_return)
