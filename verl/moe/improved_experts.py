

import torch
import torch.nn as nn
from typing import Dict, Optional
from peft import LoraConfig, get_peft_model, TaskType
import os

def load_baseline_checkpoint(checkpoint_dir):
    
    from transformers import AutoModelForCausalLM
    
    ckpt_path = None
    if os.path.exists(checkpoint_dir):
        actor_dir = os.path.join(checkpoint_dir, 'actor')
        if os.path.exists(actor_dir):
            ckpt_files = [f for f in os.listdir(actor_dir) if f.endswith('.pt')]
            if ckpt_files:
                ckpt_path = os.path.join(actor_dir, sorted(ckpt_files)[-1])
    
    if ckpt_path:
        state_dict = torch.load(ckpt_path)
        return state_dict
    else:
        return None

class ImprovedCognitiveExpert(nn.Module):

    def __init__(
        self,
        base_model: nn.Module,
        expert_id: int,
        expert_type: str,
        lora_rank: int = 64,
        baseline_checkpoint: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.expert_id = expert_id
        self.expert_type = expert_type
        
        self.expert_profiles = {
            'explore': {
                'role': 'Explore and search for target objects',
                'actions': ['go to', 'open', 'look in'],
            },
            'heat': {
                'role': 'Heat objects',
                'actions': ['heat', 'use microwave'],
            },
            'clean': {
                'role': 'Clean objects',
                'actions': ['clean', 'use sinkbasin'],
            },
            'cool': {
                'role': 'Cool objects',
                'actions': ['cool', 'use fridge'],
            },
            'navigate': {
                'role': 'Navigate to target location',
                'actions': ['go to', 'put on', 'put in'],
            },
            'recover': {
                'role': 'Error recovery',
                'actions': ['retry', 'fallback'],
            },
        }
        
        self.profile = self.expert_profiles.get(expert_type, self.expert_profiles['explore'])
        
        if baseline_checkpoint is not None:
            try:
                base_model.load_state_dict(baseline_checkpoint, strict=False)
            except Exception as e:
                pass
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            lora_dropout=0.05,
        )
        
        self.model = get_peft_model(base_model, lora_config)
        
        self._add_init_noise(expert_id)
        
    def _add_init_noise(self, expert_id: int):
        
        torch.manual_seed(42 + expert_id * 1000)
        
        for name, param in self.model.named_parameters():
            if 'lora' in name and param.requires_grad:
                noise = torch.randn_like(param) * 0.01 * (expert_id + 1)
                param.data.add_(noise)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

class ImprovedCognitiveExpertMixture(nn.Module):

    def __init__(
        self,
        base_model: nn.Module,
        num_experts: int = 6,
        lora_rank: int = 64,
        baseline_checkpoint: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        
        expert_types = ['explore', 'heat', 'clean', 'cool', 'navigate', 'recover']
        
        if num_experts == 4:
            expert_types = ['explore', 'interact', 'navigate', 'recover']
        elif num_experts == 8:
            expert_types = ['explore', 'heat', 'clean', 'cool', 'navigate', 'put', 'take', 'recover']
        
        self.experts = nn.ModuleList([
            ImprovedCognitiveExpert(
                base_model=base_model,
                expert_id=i,
                expert_type=expert_types[i] if i < len(expert_types) else 'general',
                lora_rank=lora_rank,
                baseline_checkpoint=baseline_checkpoint,
            )
            for i in range(num_experts)
        ])
        
    def forward(self, input_ids, attention_mask=None, phase_probs=None, **kwargs):
        
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
        
        batch_size = input_ids.shape[0]
        if expert_outputs:
            seq_len = expert_outputs[0][1].logits.shape[1]
            vocab_size = expert_outputs[0][1].logits.shape[2]
        else:
            seq_len, vocab_size = 1, 32000
            
        mixed_logits = torch.zeros(batch_size, seq_len, vocab_size, 
                                   device=input_ids.device, dtype=torch.bfloat16)
        
        for mask, output in expert_outputs:
            mixed_logits[mask] = output.logits
        
        return type('Output', (), {'logits': mixed_logits})()

def create_improved_phase_moe(
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_experts: int = 6,
    lora_rank: int = 64,
    baseline_checkpoint_dir: Optional[str] = None,
):
    
    baseline_ckpt = None
    if baseline_checkpoint_dir:
        baseline_ckpt = load_baseline_checkpoint(baseline_checkpoint_dir)
    
    from transformers import AutoModelForCausalLM
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    experts = ImprovedCognitiveExpertMixture(
        base_model=base_model,
        num_experts=num_experts,
        lora_rank=lora_rank,
        baseline_checkpoint=baseline_ckpt,
    )
    
    return experts
