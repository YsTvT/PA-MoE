

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

class PaperCompliantPhaseRouter(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 1536,
        num_experts: int = 4,
        history_window: int = 5,
        tau_0: float = 2.0,
        tau_f: float = 0.5,
        anneal_steps: int = 3000,
        lambda_s: float = 0.05,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.history_window = history_window
        self.lambda_s = lambda_s
        
        self.history_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            dropout=0.1,
        )
        
        self.obs_goal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        
        self.expert_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_experts)
        )
        
        self.register_buffer('tau_0', torch.tensor(tau_0))
        self.register_buffer('tau_f', torch.tensor(tau_f))
        self.register_buffer('anneal_steps', torch.tensor(anneal_steps))
        self.register_buffer('current_step', torch.tensor(0))
        self.register_buffer('current_temperature', torch.tensor(tau_0))
        
        self.register_buffer('prev_phase_modes', torch.zeros(1, dtype=torch.long))
        
    def forward(
        self,
        obs_hidden: torch.Tensor,
        goal_hidden: torch.Tensor,
        history_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = obs_hidden.shape[0]
        
        aligned_obs, _ = self.obs_goal_attention(
            query=obs_hidden,
            key=goal_hidden,
            value=goal_hidden,
        )
        obs_repr = aligned_obs.mean(dim=1)
        
        _, (h_n, _) = self.history_lstm(history_hidden)
        hist_repr = h_n[-1]
        
        combined = torch.cat([obs_repr, hist_repr], dim=-1)
        phase_logits = self.expert_mlp(combined)
        
        temp = self.current_temperature.item()
        phase_probs = F.softmax(phase_logits / temp, dim=-1)
        
        phase_modes = torch.argmax(phase_probs, dim=-1)
        
        return phase_probs, phase_logits, phase_modes
    
    def select_expert(
        self,
        phase_probs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if deterministic or not self.training:
            expert_ids = torch.argmax(phase_probs, dim=-1)
            log_probs = torch.log(phase_probs.gather(1, expert_ids.unsqueeze(1)).squeeze(1) + 1e-10)
        else:
            expert_ids = torch.multinomial(phase_probs, num_samples=1).squeeze(1)
            log_probs = torch.log(phase_probs.gather(1, expert_ids.unsqueeze(1)).squeeze(1) + 1e-10)
        
        return expert_ids, log_probs
    
    def compute_switching_penalty(
        self,
        phase_modes_sequence: torch.Tensor,
    ) -> torch.Tensor:
        
        T = phase_modes_sequence.shape[0]
        if T < 2:
            return torch.tensor(0.0, device=phase_modes_sequence.device)
        
        z_t = phase_modes_sequence[:-1]
        z_t_plus_1 = phase_modes_sequence[1:]
        
        switches = (z_t != z_t_plus_1).float()

        avg_switches = switches.mean()
        
        loss = self.lambda_s * avg_switches
        
        return loss
    
    def update_temperature(self):
        
        self.current_step += 1
        step = self.current_step.item()
        anneal_steps = self.anneal_steps.item()
        
        if step >= anneal_steps:
            self.current_temperature.fill_(self.tau_f.item())
        else:
            progress = step / anneal_steps
            tau_0 = self.tau_0.item()
            tau_f = self.tau_f.item()
            current_tau = tau_0 - (tau_0 - tau_f) * progress
            self.current_temperature.fill_(current_tau)
        
        return self.current_temperature.item()
    
    def get_statistics(self) -> dict:
        
        return {
            'router/temperature': self.current_temperature.item(),
            'router/step': self.current_step.item(),
            'router/lambda_s': self.lambda_s,
        }

def compute_router_policy_gradient_loss(
    log_probs_router: torch.Tensor,
    trajectory_return: torch.Tensor,
) -> torch.Tensor:
    
    loss = -(log_probs_router.sum() * trajectory_return)
    
    return loss

if __name__ == "__main__":
    print("Paper-Compliant Phase-Aware Router Test")
    print("=" * 60)
    
    router = PaperCompliantPhaseRouter(
        hidden_dim=1536,
        num_experts=4,
        history_window=5,
        tau_0=2.0,
        tau_f=0.5,
        anneal_steps=3000,
        lambda_s=0.05,
    )
    
    batch_size = 4
    obs_hidden = torch.randn(batch_size, 10, 1536)
    goal_hidden = torch.randn(batch_size, 5, 1536)
    history_hidden = torch.randn(batch_size, 5, 1536)
    
    phase_probs, phase_logits, phase_modes = router(obs_hidden, goal_hidden, history_hidden)
    
    router.train()
    expert_ids_train, log_probs_train = router.select_expert(phase_probs, deterministic=False)
    
    router.eval()
    expert_ids_eval, log_probs_eval = router.select_expert(phase_probs, deterministic=True)
    
    T = 20
    phase_modes_seq = torch.randint(0, 4, (T, batch_size))
    switch_loss = router.compute_switching_penalty(phase_modes_seq)
    
    temps = []
    for step in range(5):
        temp = router.update_temperature()
        temps.append(temp)
    
    log_probs_traj = torch.randn(10)
    traj_return = torch.tensor(5.0)
    router_loss = compute_router_policy_gradient_loss(log_probs_traj, traj_return)
