

import torch
import torch.nn as nn
from typing import Dict, Optional
from peft import LoraConfig, get_peft_model, TaskType

class CognitiveExpert(nn.Module):

    def __init__(
        self,
        base_model: nn.Module,
        expert_id: int,
        expert_type: str,  # 'explore', 'interact', 'navigate', 'recover'
        lora_rank: int = 16,
    ):
        super().__init__()
        
        self.expert_id = expert_id
        self.expert_type = expert_type
        
        self.expert_profiles = {
            'explore': {
                'role': 'Systematically explore unknown areas and search for target objects',
                'bias': 'Systematic traversal, avoid repetition, prioritize high-probability locations',
                'preferred_actions': ['go to', 'open', 'look in', 'examine'],
            },
            'interact': {
                'role': 'Physical interaction to change object states',
                'bias': 'Understand physical causality, use tools correctly, verify operations',
                'preferred_actions': ['take', 'heat', 'clean', 'cool', 'use'],
            },
            'navigate': {
                'role': 'Carry objects to target location',
                'bias': 'Shortest path, target matching, completion judgment',
                'preferred_actions': ['go to', 'put on', 'put in'],
            },
            'recover': {
                'role': 'Detect failures and correct strategy',
                'bias': 'Identify failure causes, retreat to safe state, try alternatives',
                'preferred_actions': ['put down', 're-observe', 'change path'],
            },
        }
        
        self.profile = self.expert_profiles[expert_type]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            lora_dropout=0.05,
        )
        
        self.model = get_peft_model(base_model, lora_config)
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
    
    def get_action_bias(self, action_text: str) -> float:
        
        action_lower = action_text.lower()
        
        bias = 0.0
        for preferred in self.profile['preferred_actions']:
            if preferred in action_lower:
                bias += 0.2
        
        return bias

class CognitiveExpertMixture(nn.Module):

    def __init__(
        self,
        base_model: nn.Module,
        lora_rank: int = 16,
    ):
        super().__init__()
        
        self.experts = nn.ModuleList([
            CognitiveExpert(base_model, 0, 'explore', lora_rank),
            CognitiveExpert(base_model, 1, 'interact', lora_rank),
            CognitiveExpert(base_model, 2, 'navigate', lora_rank),
            CognitiveExpert(base_model, 3, 'recover', lora_rank),
        ])
        
        self.num_experts = 4
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        phase_probs: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        
        if phase_probs is not None:
            primary_expert = torch.argmax(phase_probs, dim=-1)
        else:
            primary_expert = torch.zeros(input_ids.shape[0], dtype=torch.long)
        
        expert_outputs = []
        
        for expert_id in range(self.num_experts):
            mask = (primary_expert == expert_id)
            if mask.sum() == 0:
                continue
            
            expert_input_ids = input_ids[mask]
            expert_attention_mask = attention_mask[mask] if attention_mask is not None else None
            
            output = self.experts[expert_id](
                input_ids=expert_input_ids,
                attention_mask=expert_attention_mask,
                **kwargs,
            )
            
            expert_outputs.append((mask, output))
        
        batch_size, seq_len, vocab_size = input_ids.shape[0], output.logits.shape[1], output.logits.shape[2]
        mixed_logits = torch.zeros(batch_size, seq_len, vocab_size, 
                                   device=input_ids.device, dtype=output.logits.dtype)
        
        for mask, output in expert_outputs:
            mixed_logits[mask] = output.logits
        
        return type('Output', (), {'logits': mixed_logits})()
    
    def get_expert_by_phase(self, phase_id: int):
        
        return self.experts[phase_id]

def compute_expert_mutual_exclusion_loss(
    expert_outputs: List[torch.Tensor],
    phase_probs: torch.Tensor,
    threshold: float = 0.3,
    penalty_weight: float = 0.1,
) -> torch.Tensor:
    
    num_experts = len(expert_outputs)
    total_loss = 0.0
    num_pairs = 0
    
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            p_i = F.softmax(expert_outputs[i], dim=-1)
            p_j = F.softmax(expert_outputs[j], dim=-1)
            
            kl_div = F.kl_div(
                p_j.log(),
                p_i,
                reduction='batchmean',
            )
            
            co_activation = (phase_probs[:, i] * phase_probs[:, j]).mean()
            
            if kl_div < threshold:
                loss = (threshold - kl_div) * co_activation
                total_loss += loss
                num_pairs += 1
    
    if num_pairs > 0:
        total_loss = total_loss / num_pairs
    
    return total_loss * penalty_weight

def compute_load_balance_loss(
    phase_probs: torch.Tensor,
    target_uniform: float = 0.25,
    penalty_weight: float = 0.01,
) -> torch.Tensor:
    
    avg_usage = phase_probs.mean(dim=0)
    
    uniform = torch.ones_like(avg_usage) * target_uniform
    deviation = ((avg_usage - uniform) ** 2).mean()
    
    return deviation * penalty_weight
