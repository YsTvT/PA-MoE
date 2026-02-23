

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from transformers import AutoModelForCausalLM

from .phase_router import PhaseAwareRouter, auto_label_phases, compute_phase_alignment_reward
from .cognitive_experts import CognitiveExpertMixture, compute_expert_mutual_exclusion_loss, compute_load_balance_loss
from .hierarchical_critic import HierarchicalCritic

class PhaseAwareMoEActor(nn.Module):

    def __init__(
        self,
        base_model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        lora_rank: int = 16,
        device: str = "cuda",
    ):
        super().__init__()
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        hidden_dim = base_model.config.hidden_size
        
        self.router = PhaseAwareRouter(
            hidden_dim=hidden_dim,
            num_phases=4,
        )
        
        self.experts = CognitiveExpertMixture(
            base_model=base_model,
            lora_rank=lora_rank,
        )
        
        self.hidden_dim = hidden_dim
        self.num_phases = 4
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        obs_hidden: Optional[torch.Tensor] = None,
        goal_hidden: Optional[torch.Tensor] = None,
        history_hidden: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        
        if obs_hidden is not None and goal_hidden is not None and history_hidden is not None:
            phase_probs, transition_signal = self.router(
                obs_hidden=obs_hidden,
                goal_hidden=goal_hidden,
                history_hidden=history_hidden,
            )
        else:
            batch_size = input_ids.shape[0]
            phase_probs = torch.ones(batch_size, 4, device=input_ids.device) * 0.25
            transition_signal = torch.zeros(batch_size, device=input_ids.device)
        
        output = self.experts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            phase_probs=phase_probs,
            **kwargs,
        )
        
        return type('PhaseAwareMoEOutput', (), {
            'logits': output.logits,
            'phase_probs': phase_probs,
            'transition_signal': transition_signal,
            'selected_phase': torch.argmax(phase_probs, dim=-1),
        })()

def create_phase_aware_moe(
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    lora_rank: int = 16,
) -> Tuple[PhaseAwareMoEActor, HierarchicalCritic]:
    
    actor = PhaseAwareMoEActor(
        base_model_path=model_path,
        lora_rank=lora_rank,
    )
    
    from transformers import AutoModelForTokenClassification, AutoConfig
    
    critic_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    critic_config.num_labels = 1
    
    base_critic = AutoModelForTokenClassification.from_pretrained(
        model_path,
        config=critic_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    critic = HierarchicalCritic(
        base_critic=base_critic,
        num_phases=4,
        hidden_dim=actor.hidden_dim,
    )
    
    return actor, critic
