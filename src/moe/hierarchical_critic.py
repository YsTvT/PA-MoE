
import torch
import torch.nn as nn
from typing import Optional

class HierarchicalCritic(nn.Module):

    def __init__(
        self,
        base_critic: nn.Module,
        num_phases: int = 4,
        hidden_dim: int = 1536,
    ):
        super().__init__()
        
        self.num_phases = num_phases

        self.global_critic = base_critic

        self.phase_critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_phases)
        ])
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        phase_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):

        global_output = self.global_critic(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        
        V_global = global_output.logits
        if V_global.dim() == 3:
            V_global = V_global.squeeze(-1)

        if phase_ids is not None:
            hidden_states = global_output.hidden_states[-1]
            state_repr = hidden_states.mean(dim=1)

            batch_size = input_ids.shape[0]
            V_phase = torch.zeros(batch_size, 1, device=input_ids.device)
            
            for phase_id in range(self.num_phases):
                mask = (phase_ids == phase_id)
                if mask.sum() == 0:
                    continue
                
                phase_value = self.phase_critics[phase_id](state_repr[mask])
                V_phase[mask] = phase_value

            V_phase = V_phase.expand(-1, V_global.shape[1])
        else:
            V_phase = 0.0

        V_total = V_global + V_phase
        
        return type('Output', (), {'logits': V_total})()

def compute_hierarchical_advantage(
    rewards: torch.Tensor,
    values_global: torch.Tensor,
    values_phase: torch.Tensor,
    phase_masks: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    lambda_phase: float = 0.5,
):

    from verl.trainer.ppo.core_algos import compute_gae_advantage_return
    
    A_global, returns_global = compute_gae_advantage_return(
        token_level_rewards=rewards,
        values=values_global,
        response_mask=torch.ones_like(rewards),
        gamma=torch.tensor(gamma),
        lam=torch.tensor(lam),
    )

    A_phase = torch.zeros_like(A_global)
    
    for phase_id in range(4):
        phase_mask = phase_masks[:, :, phase_id]
        
        if phase_mask.sum() == 0:
            continue

        phase_rewards = rewards * phase_mask
        phase_values = values_phase * phase_mask
        
        A_phase_k, _ = compute_gae_advantage_return(
            token_level_rewards=phase_rewards,
            values=phase_values,
            response_mask=phase_mask,
            gamma=torch.tensor(gamma),
            lam=torch.tensor(lam),
        )
        
        A_phase += A_phase_k * phase_mask

    A_total = A_global + lambda_phase * A_phase
    
    return A_total, returns_global
