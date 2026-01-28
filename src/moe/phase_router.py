
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List

class PhaseAwareRouter(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 1536,
        num_phases: int = 4,
        history_len: int = 10,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_phases = num_phases
        self.history_len = history_len

        self.history_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        self.obs_goal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        self.phase_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_phases)
        )

        self.transition_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(
        self,
        obs_hidden: torch.Tensor,
        goal_hidden: torch.Tensor,
        history_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = obs_hidden.shape[0]
        
        obs_repr = obs_hidden.mean(dim=1)
        goal_repr = goal_hidden.mean(dim=1)
        
        hist_output, (h_n, c_n) = self.history_lstm(history_hidden)
        hist_repr = h_n[-1]

        aligned_obs, _ = self.obs_goal_attention(
            query=obs_repr.unsqueeze(1),
            key=goal_repr.unsqueeze(1),
            value=goal_repr.unsqueeze(1),
        )
        aligned_obs = aligned_obs.squeeze(1)

        hist_repr_padded = F.pad(hist_repr, (0, self.hidden_dim // 2))
        fused = torch.cat([aligned_obs, hist_repr_padded], dim=-1)

        phase_logits = self.phase_classifier(fused)
        phase_probs = F.softmax(phase_logits, dim=-1)

        transition_logit = self.transition_detector(fused)
        transition_signal = torch.sigmoid(transition_logit).squeeze(-1)
        
        return phase_probs, transition_signal
    
    def detect_phase_transition(
        self,
        current_phase_probs: torch.Tensor,
        prev_phase_probs: torch.Tensor,
        transition_signal: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        
        current_phase = torch.argmax(current_phase_probs, dim=-1)
        prev_phase = torch.argmax(prev_phase_probs, dim=-1)
        
        phase_changed = (current_phase != prev_phase).float()
        high_transition = (transition_signal > threshold).float()
        
        transition = phase_changed * high_transition
        
        return transition.bool()

def auto_label_phases(trajectory: List[Dict]) -> List[int]:
    
    phases = []
    
    has_target = False
    processed = False
    consecutive_fails = 0
    
    for i, step in enumerate(trajectory):
        action = step['action'].lower()
        reward = step.get('reward', 0)

        if reward < 0 or 'nothing happens' in step.get('observation', ''):
            consecutive_fails += 1
        else:
            consecutive_fails = 0

        if consecutive_fails >= 3:
            phases.append(3)
            continue

        if any(verb in action for verb in ['take', 'heat', 'clean', 'cool', 'put in']):
            if 'take' in action:
                has_target = True
                phases.append(1)
            elif any(verb in action for verb in ['heat', 'clean', 'cool']):
                processed = True
                phases.append(1)
            else:
                phases.append(1)
            continue

        if has_target and processed:
            if 'go to' in action or 'put' in action:
                phases.append(2)
                continue

        phases.append(0)
    
    return phases

def compute_phase_alignment_reward(
    phase_probs: torch.Tensor,
    expert_id: int,
    outcome: float,
    bonus: float = 0.1,
) -> torch.Tensor:

    router_confidence = phase_probs[:, expert_id]

    if outcome > 0:
        alignment_reward = router_confidence * bonus
    else:
        alignment_reward = -router_confidence * bonus
    
    return alignment_reward
