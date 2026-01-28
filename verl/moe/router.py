

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

class TopKGate(nn.Module):

    def __init__(
        self, 
        state_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.0,
        use_softmax: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std
        self.use_softmax = use_softmax
        
        self.gate = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.ReLU(),
            nn.Linear(state_dim * 2, num_experts)
        )
        
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('total_calls', torch.tensor(0))
        
    def forward(
        self, 
        state: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        original_shape = state.shape
        if len(state.shape) == 3:
            batch_size, seq_len, state_dim = state.shape
            state = state.reshape(-1, state_dim)
        else:
            batch_size = state.shape[0]
            seq_len = 1
            
        logits = self.gate(state)
        
        if training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
            
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        
        if self.use_softmax:
            top_k_gates = F.softmax(top_k_logits, dim=-1)
        else:
            top_k_gates = torch.sigmoid(top_k_logits)
            top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-8)
        
        gates = torch.zeros_like(logits)
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        load_balancing_loss = self._compute_load_balancing_loss(gates, training)
        
        if training:
            self._update_usage_stats(gates)
            
        if len(original_shape) == 3:
            gates = gates.reshape(batch_size, seq_len, self.num_experts)
            top_k_indices = top_k_indices.reshape(batch_size, seq_len, self.top_k)
        
        return gates, top_k_indices, load_balancing_loss
    
    def _compute_load_balancing_loss(
        self, 
        gates: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        
        if not training:
            return torch.tensor(0.0, device=gates.device)
            
        avg_gates = gates.mean(dim=0)
        
        load_variance = torch.var(avg_gates)
        
        return load_variance
    
    def _update_usage_stats(self, gates: torch.Tensor):
        
        batch_usage = (gates > 0).float().sum(dim=0)
        self.expert_usage += batch_usage
        self.total_calls += gates.shape[0]
        
    def get_usage_stats(self) -> Dict[str, torch.Tensor]:
        
        if self.total_calls == 0:
            return {
                'usage_fraction': torch.zeros(self.num_experts),
                'usage_std': torch.tensor(0.0),
            }
            
        usage_fraction = self.expert_usage / self.total_calls
        return {
            'usage_fraction': usage_fraction,
            'usage_std': torch.std(usage_fraction),
            'usage_min': torch.min(usage_fraction),
            'usage_max': torch.max(usage_fraction),
        }
    
    def reset_usage_stats(self):
        
        self.expert_usage.zero_()
        self.total_calls.zero_()

class MoERouter(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.0,
        state_pooling: str = 'mean',  # 'mean', 'last', 'first'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.state_pooling = state_pooling
        
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        self.gate = TopKGate(
            state_dim=hidden_dim // 2,
            num_experts=num_experts,
            top_k=top_k,
            noise_std=noise_std,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if self.state_pooling == 'mean':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1)
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-8)
                state_repr = sum_hidden / sum_mask
            else:
                state_repr = hidden_states.mean(dim=1)
        elif self.state_pooling == 'last':
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                state_repr = hidden_states[batch_indices, seq_lengths]
            else:
                state_repr = hidden_states[:, -1]
        elif self.state_pooling == 'first':
            state_repr = hidden_states[:, 0]
        else:
            raise ValueError(f"Unknown state_pooling: {self.state_pooling}")
        
        state_encoded = self.state_encoder(state_repr)
        
        gates, top_k_indices, load_balancing_loss = self.gate(
            state_encoded, training=training
        )
        
        return gates, top_k_indices, load_balancing_loss
    
    def get_usage_stats(self) -> Dict[str, torch.Tensor]:
        
        return self.gate.get_usage_stats()
    
    def reset_usage_stats(self):
        
        self.gate.reset_usage_stats()
