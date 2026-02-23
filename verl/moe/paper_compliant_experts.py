

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from peft import LoraConfig, get_peft_model, TaskType

class PaperCompliantLoRAExpert(nn.Module):

    def __init__(
        self,
        base_model: nn.Module,
        expert_id: int,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        target_modules: List[str] = None,
    ):
        super().__init__()
        
        self.expert_id = expert_id
        self.lora_rank = lora_rank
        
        if target_modules is None:
            target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        
        self.model = get_peft_model(base_model, lora_config)
        
        lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.param_overhead = lora_params / total_params
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def get_info(self):
        
        return {
            'expert_id': self.expert_id,
            'lora_rank': self.lora_rank,
            'param_overhead': f"{self.param_overhead * 100:.2f}%",
        }

class PaperCompliantExpertMixture(nn.Module):

    def __init__(
        self,
        base_model: nn.Module,
        num_experts: int = 4,
        lora_rank: int = 32,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.lora_rank = lora_rank
        
        for param in base_model.parameters():
            param.requires_grad = False
        
        self.experts = nn.ModuleList([
            PaperCompliantLoRAExpert(
                base_model=base_model,
                expert_id=i,
                lora_rank=lora_rank,
            )
            for i in range(num_experts)
        ])
    
    def forward_single_expert(
        self,
        expert_id: int,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        
        return self.experts[expert_id](
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def forward_with_routing(
        self,
        expert_ids: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        sample_output = self.experts[0](
            input_ids=input_ids[:1],
            attention_mask=attention_mask[:1] if attention_mask is not None else None,
            **kwargs
        )
        
        if hasattr(sample_output, 'logits'):
            output_shape = sample_output.logits.shape
            all_logits = torch.zeros(batch_size, *output_shape[1:], device=device)
        else:
            raise NotImplementedError("Only supports models with .logits output")
        
        for expert_id in range(self.num_experts):
            mask = (expert_ids == expert_id)
            if not mask.any():
                continue
            
            expert_input_ids = input_ids[mask]
            expert_attention_mask = attention_mask[mask] if attention_mask is not None else None
            
            expert_output = self.experts[expert_id](
                input_ids=expert_input_ids,
                attention_mask=expert_attention_mask,
                **kwargs
            )
            
            all_logits[mask] = expert_output.logits
        
        return type('Output', (), {
            'logits': all_logits,
        })()

def compute_diversity_loss(
    expert_outputs: List[torch.Tensor],
    phase_probs: torch.Tensor,
    tau_div: float = 0.1,
    lambda_div: float = 0.01,
) -> torch.Tensor:
    
    num_experts = len(expert_outputs)
    total_loss = 0.0
    num_pairs = 0
    
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            logits_i = expert_outputs[i]
            logits_j = expert_outputs[j]
            
            probs_i = F.softmax(logits_i, dim=-1)
            probs_j = F.softmax(logits_j, dim=-1)
            
            kl_div = F.kl_div(
                probs_j.log(),
                probs_i,
                reduction='batchmean',
            )
            
            co_activation = (phase_probs[:, i] * phase_probs[:, j]).mean()
            
            if kl_div < tau_div:
                penalty = (tau_div - kl_div) * co_activation
                total_loss += penalty
                num_pairs += 1
    
    if num_pairs > 0:
        total_loss = total_loss / num_pairs
    
    return total_loss * lambda_div

def compute_balance_loss(
    phase_probs: torch.Tensor,
    lambda_bal: float = 0.001,
) -> torch.Tensor:
    
    num_experts = phase_probs.shape[1]
    
    f_k = phase_probs.mean(dim=0)
    
    target = 1.0 / num_experts
    
    loss = ((f_k - target) ** 2).sum()
    
    return loss * lambda_bal

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
    )
    
    expert_mixture = PaperCompliantExpertMixture(
        base_model=base_model,
        num_experts=4,
        lora_rank=32,
    )
    
    input_ids = torch.randint(0, 1000, (2, 10))
    output = expert_mixture.forward_single_expert(
        expert_id=0,
        input_ids=input_ids,
    )
    
    expert_ids = torch.tensor([0, 1, 2, 3, 0, 1])
    input_ids = torch.randint(0, 1000, (6, 10))
    output = expert_mixture.forward_with_routing(
        expert_ids=expert_ids,
        input_ids=input_ids,
    )
    
    expert_outputs = [torch.randn(4, 10, 1000) for _ in range(4)]
    phase_probs = F.softmax(torch.randn(4, 4), dim=-1)
    div_loss = compute_diversity_loss(expert_outputs, phase_probs, tau_div=0.1, lambda_div=0.01)
    
    bal_loss = compute_balance_loss(phase_probs, lambda_bal=0.001)
