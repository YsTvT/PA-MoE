

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    agg_loss,
)

def compute_moe_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    gates: torch.Tensor,
    expert_masks: Optional[torch.Tensor] = None,
    cliprange: float = 0.2,
    cliprange_low: Optional[float] = None,
    cliprange_high: Optional[float] = None,
    loss_agg_mode: str = 'mean',
    use_winner_take_all: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    
    ratio = torch.exp(log_prob - old_log_prob)
    
    if use_winner_take_all and expert_masks is not None:
        expert_assignment = torch.argmax(gates, dim=-1)
        
        expert_weight = torch.gather(gates, 1, expert_assignment.unsqueeze(1)).squeeze(1)
        expert_weight = expert_weight.unsqueeze(1)
        
        weighted_advantages = advantages * expert_weight
    else:
        weighted_advantages = advantages
    
    pg_loss1 = -weighted_advantages * ratio
    pg_loss2 = -weighted_advantages * torch.clamp(
        ratio, 
        1.0 - cliprange_low,
        1.0 + cliprange_high
    )
    pg_loss = torch.max(pg_loss1, pg_loss2)
    
    pg_loss = agg_loss(loss_mat=pg_loss, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    
    with torch.no_grad():
        pg_clipfrac = (
            ((ratio < 1.0 - cliprange_low) | (ratio > 1.0 + cliprange_high)).float() * response_mask
        ).sum() / response_mask.sum()
        
        pg_clipfrac_lower = (
            (ratio < 1.0 - cliprange_low).float() * response_mask
        ).sum() / response_mask.sum()
        
        ppo_kl = ((ratio - 1) - torch.log(ratio)) * response_mask
        ppo_kl = ppo_kl.sum() / response_mask.sum()
    
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower

def compute_moe_value_loss(
    vpreds: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    response_mask: torch.Tensor,
    gates: Optional[torch.Tensor] = None,
    expert_idx: Optional[int] = None,
    cliprange_value: float = 0.2,
    loss_agg_mode: str = 'mean',
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if gates is not None and expert_idx is not None:
        expert_weight = gates[:, expert_idx].unsqueeze(1)
    else:
        expert_weight = 1.0
    
    vpredclipped = values + torch.clamp(vpreds - values, -cliprange_value, cliprange_value)
    
    vf_loss1 = (vpreds - returns) ** 2
    vf_loss2 = (vpredclipped - returns) ** 2
    vf_loss = torch.max(vf_loss1, vf_loss2)
    
    vf_loss = vf_loss * expert_weight
    
    vf_loss = agg_loss(loss_mat=vf_loss, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    
    with torch.no_grad():
        vf_clipfrac = (
            (torch.abs(vpreds - values) > cliprange_value).float() * response_mask
        ).sum() / response_mask.sum()
    
    return vf_loss, vf_clipfrac

def compute_expert_advantages(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gates: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    per_expert_critics: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if per_expert_critics:
        gates_expanded = gates.unsqueeze(1)
        values_weighted = (values * gates_expanded).sum(dim=-1)
    else:
        values_weighted = values
    
    advantages, returns = compute_gae_advantage_return(
        token_level_rewards=token_level_rewards,
        values=values_weighted,
        response_mask=response_mask,
        gamma=torch.tensor(gamma),
        lam=torch.tensor(lam),
    )
    
    return advantages, returns

def compute_load_balancing_loss(
    gates: torch.Tensor,
    expert_usage: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    
    avg_gates = gates.mean(dim=0)
    load_variance = torch.var(avg_gates)
    
    uniform_target = 1.0 / num_experts
    deviation = torch.mean((avg_gates - uniform_target) ** 2)
    
    load_loss = load_variance + 0.1 * deviation
    
    return load_loss

def select_expert_winners(
    expert_q_values: torch.Tensor,
    gates: torch.Tensor,
    use_routing_weights: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    
    if use_routing_weights:
        weighted_q = expert_q_values * gates
        winner_indices = torch.argmax(weighted_q, dim=-1)
    else:
        winner_indices = torch.argmax(expert_q_values, dim=-1)
    
    batch_size, num_experts = expert_q_values.shape
    winner_mask = torch.zeros_like(expert_q_values)
    winner_mask.scatter_(1, winner_indices.unsqueeze(1), 1.0)
    
    winner_counts = torch.bincount(winner_indices, minlength=num_experts)
    winner_distribution = winner_counts.float() / batch_size
    
    stats = {
        'winner_distribution': winner_distribution.cpu().numpy(),
        'winner_entropy': -torch.sum(
            winner_distribution * torch.log(winner_distribution + 1e-10)
        ).item(),
    }
    
    return winner_mask, stats
