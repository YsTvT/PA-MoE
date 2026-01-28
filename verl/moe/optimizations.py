

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class EnhancedPhaseRouter(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 1536,
        num_phases: int = 6,
        history_len: int = 10,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_phases = num_phases
        
        self.history_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
        )
        
        self.obs_goal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=12,
            dropout=0.1,
            batch_first=True,
        )
        
        self.complexity_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3),
        )
        
        self.phase_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_phases)
        )
        
        self.enable_multi_expert = True
        
    def forward(
        self,
        obs_hidden: torch.Tensor,
        goal_hidden: torch.Tensor,
        history_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = obs_hidden.shape[0]
        
        obs_repr = obs_hidden.mean(dim=1)
        goal_repr = goal_hidden.mean(dim=1)
        
        complexity_logits = self.complexity_detector(goal_repr)
        complexity_probs = F.softmax(complexity_logits, dim=-1)
        complexity_level = torch.argmax(complexity_probs, dim=-1)
        
        hist_output, (h_n, c_n) = self.history_lstm(history_hidden)
        hist_repr = h_n[-1]
        
        aligned_obs, attn_weights = self.obs_goal_attention(
            query=obs_repr.unsqueeze(1),
            key=goal_repr.unsqueeze(1),
            value=goal_repr.unsqueeze(1),
        )
        aligned_obs = aligned_obs.squeeze(1)
        
        fused = torch.cat([aligned_obs, goal_repr, hist_repr], dim=-1)
        
        phase_logits = self.phase_classifier(fused)
        phase_probs = F.softmax(phase_logits, dim=-1)
        
        k = torch.where(
            complexity_level == 2,
            torch.tensor(2, device=obs_hidden.device),
            torch.tensor(1, device=obs_hidden.device)
        )
        
        top_k_values, top_k_experts = torch.topk(phase_probs, k=2, dim=-1)
        
        return phase_probs, complexity_probs, top_k_experts

class AdaptiveCritic(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 1536,
        num_phases: int = 6,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_phases = num_phases
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.global_value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.phase_value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_phases)
        ])
        
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(
        self,
        state_hidden: torch.Tensor,
        current_phase: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = state_hidden.shape[0]
        
        features = self.shared_encoder(state_hidden.mean(dim=1))
        
        global_value = self.global_value_head(features).squeeze(-1)
        
        if current_phase is not None:
            phase_values = []
            for i in range(batch_size):
                phase_id = current_phase[i].item()
                phase_val = self.phase_value_heads[phase_id](features[i:i+1])
                phase_values.append(phase_val)
            phase_value = torch.cat(phase_values, dim=0).squeeze(-1)
        else:
            phase_value = torch.zeros(batch_size, device=state_hidden.device)
        
        reward_pred = self.reward_predictor(features).squeeze(-1)
        
        return global_value, phase_value, reward_pred

class MultiExpertCoordinator(nn.Module):

    def __init__(self, num_experts: int = 6):
        super().__init__()
        self.num_experts = num_experts
        
        self.register_buffer(
            'coordination_matrix',
            self._build_coordination_matrix()
        )
        
    def _build_coordination_matrix(self):
        
        matrix = torch.zeros(6, 6)
        
        collaborations = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (2, 4),
            (3, 4),
            (0, 4),
            (4, 0),
        ]
        
        for i, j in collaborations:
            matrix[i, j] = 1.0
            
        return matrix
    
    def select_expert_sequence(
        self,
        phase_probs: torch.Tensor,
        complexity_level: torch.Tensor,
    ):
        
        batch_size = phase_probs.shape[0]
        
        sequences = []
        for i in range(batch_size):
            if complexity_level[i] == 2:
                top2 = torch.topk(phase_probs[i], k=2).indices
                if self.coordination_matrix[top2[0], top2[1]] > 0:
                    sequences.append(top2.tolist())
                else:
                    sequences.append([top2[0].item()])
            else:
                sequences.append([torch.argmax(phase_probs[i]).item()])
        
        return sequences

def create_optimized_modules(
    hidden_dim: int = 1536,
    num_experts: int = 6,
):
    
    router = EnhancedPhaseRouter(
        hidden_dim=hidden_dim,
        num_phases=num_experts,
    )
    
    critic = AdaptiveCritic(
        hidden_dim=hidden_dim,
        num_phases=num_experts,
    )
    
    coordinator = MultiExpertCoordinator(
        num_experts=num_experts,
    )
    
    return router, critic, coordinator

def compute_enhanced_loss(
    phase_probs: torch.Tensor,
    complexity_probs: torch.Tensor,
    global_value: torch.Tensor,
    phase_value: torch.Tensor,
    reward_pred: torch.Tensor,
    true_phase: torch.Tensor,
    true_complexity: torch.Tensor,
    returns: torch.Tensor,
    rewards: torch.Tensor,
    expert_usage: torch.Tensor,
):

    router_loss = F.cross_entropy(phase_probs, true_phase)
    
    complexity_loss = F.cross_entropy(complexity_probs, true_complexity)
    
    global_value_loss = F.mse_loss(global_value, returns)
    phase_value_loss = F.mse_loss(phase_value, returns)
    critic_loss = global_value_loss + 0.5 * phase_value_loss
    
    reward_loss = F.mse_loss(reward_pred, rewards)
    
    expert_usage_mean = expert_usage.mean(dim=0)
    target_usage = 1.0 / expert_usage.shape[1]
    balance_loss = F.mse_loss(
        expert_usage_mean,
        torch.full_like(expert_usage_mean, target_usage)
    )
    
    total_loss = (
        1.0 * router_loss +
        0.5 * complexity_loss +
        1.0 * critic_loss +
        0.3 * reward_loss +
        0.2 * balance_loss
    )
    
    return {
        'total': total_loss,
        'router': router_loss,
        'complexity': complexity_loss,
        'critic': critic_loss,
        'reward': reward_loss,
        'balance': balance_loss,
    }
