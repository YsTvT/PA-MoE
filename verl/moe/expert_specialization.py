

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

class WinnerTakeAll:

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.expert_wins = np.zeros(num_experts)
        self.total_competitions = 0
        
    def select_winner(
        self,
        expert_q_values: torch.Tensor,
        gates: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        
        winner_indices = torch.argmax(expert_q_values, dim=-1)
        
        batch_size = expert_q_values.shape[0]
        winner_mask = torch.zeros_like(expert_q_values)
        winner_mask.scatter_(1, winner_indices.unsqueeze(1), 1.0)
        
        winner_counts = torch.bincount(
            winner_indices, 
            minlength=self.num_experts
        ).cpu().numpy()
        self.expert_wins += winner_counts
        self.total_competitions += batch_size
        
        stats = {
            'winner_distribution': winner_counts,
            'winner_entropy': self._compute_entropy(winner_counts),
        }
        
        return winner_mask, stats
    
    def _compute_entropy(self, counts: np.ndarray) -> float:
        
        if counts.sum() == 0:
            return 0.0
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs + 1e-10))
    
    def get_stats(self) -> Dict:
        
        if self.total_competitions == 0:
            return {
                'win_rate': np.zeros(self.num_experts),
                'win_rate_std': 0.0,
            }
        
        win_rate = self.expert_wins / self.total_competitions
        return {
            'win_rate': win_rate,
            'win_rate_std': np.std(win_rate),
            'win_rate_min': np.min(win_rate),
            'win_rate_max': np.max(win_rate),
        }
    
    def reset_stats(self):
        
        self.expert_wins = np.zeros(self.num_experts)
        self.total_competitions = 0

class LoadBalancer:

    def __init__(
        self,
        num_experts: int,
        capacity_factor: float = 1.5,
        penalty_weight: float = 0.1,
    ):
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.penalty_weight = penalty_weight
        
        self.expert_counts = np.zeros(num_experts)
        self.capacity_limit = None
        
    def update_capacity(self, total_samples: int):
        
        avg_capacity = total_samples / self.num_experts
        self.capacity_limit = int(avg_capacity * self.capacity_factor)
        
    def apply_penalty(
        self,
        routing_scores: torch.Tensor,
    ) -> torch.Tensor:
        
        if self.capacity_limit is None:
            return routing_scores
            
        utilization = torch.tensor(
            self.expert_counts / max(self.capacity_limit, 1),
            device=routing_scores.device,
            dtype=routing_scores.dtype,
        )
        
        penalty = self.penalty_weight * utilization.unsqueeze(0)
        penalized_scores = routing_scores - penalty
        
        return penalized_scores
    
    def update_counts(self, assignments: torch.Tensor):
        
        counts = torch.bincount(
            assignments,
            minlength=self.num_experts,
        ).cpu().numpy()
        self.expert_counts += counts
        
    def reset_counts(self):
        
        self.expert_counts = np.zeros(self.num_experts)
        
    def get_stats(self) -> Dict:
        
        if self.expert_counts.sum() == 0:
            return {
                'load_distribution': self.expert_counts,
                'load_imbalance': 0.0,
            }
        
        mean_load = self.expert_counts.mean()
        std_load = self.expert_counts.std()
        load_imbalance = std_load / (mean_load + 1e-8)
        
        return {
            'load_distribution': self.expert_counts,
            'load_imbalance': load_imbalance,
            'load_min': self.expert_counts.min(),
            'load_max': self.expert_counts.max(),
        }

class GradientCompetition:

    def __init__(
        self,
        num_experts: int,
        competition_weight: float = 0.05,
    ):
        self.num_experts = num_experts
        self.competition_weight = competition_weight
        
        self.expert_gradients = {}
        
    def compute_competition_loss(
        self,
        gates: torch.Tensor,
        expert_params_list: List[torch.nn.Parameter],
    ) -> torch.Tensor:
        
        if len(expert_params_list) != self.num_experts:
            raise ValueError(
                f"Expected {self.num_experts} expert param groups, "
                f"got {len(expert_params_list)}"
            )
        
        expert_grads = []
        for params in expert_params_list:
            grads = []
            for p in params:
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
            if len(grads) > 0:
                expert_grad = torch.cat(grads)
                expert_grads.append(expert_grad)
            else:
                expert_grads.append(None)
        
        competition_loss = 0.0
        num_pairs = 0
        
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                if expert_grads[i] is None or expert_grads[j] is None:
                    continue
                
                gate_product = (gates[:, i] * gates[:, j]).mean()
                
                if gate_product < 1e-6:
                    continue
                
                cos_sim = F.cosine_similarity(
                    expert_grads[i].unsqueeze(0),
                    expert_grads[j].unsqueeze(0),
                    dim=1,
                )
                
                competition_loss += gate_product * cos_sim
                num_pairs += 1
        
        if num_pairs > 0:
            competition_loss = competition_loss / num_pairs
        else:
            competition_loss = torch.tensor(0.0, device=gates.device)
        
        return competition_loss * self.competition_weight
    
    def compute_simple_competition_loss(
        self,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        
        competition_loss = 0.0
        num_pairs = 0
        
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                overlap = (gates[:, i] * gates[:, j]).mean()
                competition_loss += overlap ** 2
                num_pairs += 1
        
        if num_pairs > 0:
            competition_loss = competition_loss / num_pairs
        
        return competition_loss * self.competition_weight
