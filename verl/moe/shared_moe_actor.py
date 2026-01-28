

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

class ExpertHead(nn.Module):

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        return self.lm_head(hidden_states)

class SharedMoEActor(nn.Module):

    def __init__(
        self,
        base_model: nn.Module,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_dim: int = None,
        vocab_size: int = None,
        router_noise_std: float = 0.1,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        
        self.base_model = base_model
        
        if hidden_dim is None:
            hidden_dim = base_model.config.hidden_size
        if vocab_size is None:
            vocab_size = base_model.config.vocab_size
            
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        if hasattr(base_model, 'lm_head'):
            del base_model.lm_head
        
        self.expert_heads = nn.ModuleList([
            ExpertHead(hidden_dim, vocab_size) 
            for _ in range(num_experts)
        ])
        
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        self.noise_std = router_noise_std
        
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('total_calls', torch.tensor(0))
        
    def route(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-8)
            state_repr = sum_hidden / sum_mask
        else:
            state_repr = hidden_states.mean(dim=1)
        
        logits = self.router(state_repr)
        
        if training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        gates = torch.zeros_like(logits)
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        load_loss = torch.tensor(0.0, device=gates.device)
        if training:
            avg_gates = gates.mean(dim=0)
            load_loss = torch.var(avg_gates) * self.load_balance_weight
            
            batch_usage = (gates > 0).float().sum(dim=0)
            self.expert_usage += batch_usage
            self.total_calls += gates.shape[0]
        
        return gates, top_k_indices, load_loss
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_dict: bool = True,
        temperature: float = 1.0,
        **kwargs,
    ):
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            output_hidden_states=True,
            **kwargs,
        )
        
        hidden_states = outputs.hidden_states[-1]
        
        gates, top_k_indices, load_loss = self.route(
            hidden_states,
            attention_mask=attention_mask,
            training=self.training,
        )
        
        batch_size, seq_len, _ = hidden_states.shape
        
        unique_experts = torch.unique(top_k_indices).tolist()
        
        expert_logits = []
        for i in range(self.num_experts):
            if i in unique_experts or not self.training:
                logits = self.expert_heads[i](hidden_states)
                expert_logits.append(logits)
            else:
                expert_logits.append(None)
        
        mixed_logits = self._mix_expert_logits(expert_logits, gates)
        
        mixed_logits = mixed_logits / temperature
        
        return MoEOutput(
            logits=mixed_logits,
            hidden_states=hidden_states,
            gates=gates,
            top_k_indices=top_k_indices,
            load_loss=load_loss,
            expert_logits=expert_logits,
        )
    
    def _mix_expert_logits(
        self,
        expert_logits: List[Optional[torch.Tensor]],
        gates: torch.Tensor,
    ) -> torch.Tensor:
        
        valid_logits = []
        valid_gates = []
        
        for i, logits in enumerate(expert_logits):
            if logits is not None:
                valid_logits.append(logits)
                valid_gates.append(gates[:, i])
        
        logits_stack = torch.stack(valid_logits, dim=0)
        gates_stack = torch.stack(valid_gates, dim=0)
        
        gates_expanded = gates_stack.unsqueeze(-1).unsqueeze(-1)
        
        mixed = (logits_stack * gates_expanded).sum(dim=0)
        
        return mixed
    
    def get_expert_params(self, expert_idx: int) -> List[nn.Parameter]:
        
        return list(self.expert_heads[expert_idx].parameters())
    
    def get_router_params(self) -> List[nn.Parameter]:
        
        return list(self.router.parameters())
    
    def get_shared_params(self) -> List[nn.Parameter]:
        
        return list(self.base_model.parameters())
    
    def get_usage_stats(self) -> Dict:
        
        if self.total_calls == 0:
            return {
                'usage_fraction': torch.zeros(self.num_experts),
                'usage_variance': 0.0,
            }
        
        usage_fraction = self.expert_usage / self.total_calls
        return {
            'usage_fraction': usage_fraction.cpu(),
            'usage_variance': torch.var(usage_fraction).item(),
            'usage_min': torch.min(usage_fraction).item(),
            'usage_max': torch.max(usage_fraction).item(),
        }
    
    def reset_usage_stats(self):
        
        self.expert_usage.zero_()
        self.total_calls.zero_()

@dataclass
class MoEOutput:
    
    logits: torch.Tensor
    hidden_states: torch.Tensor
    gates: torch.Tensor
    top_k_indices: torch.Tensor
    load_loss: torch.Tensor
    expert_logits: List[Optional[torch.Tensor]]

def create_moe_actor(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_experts: int = 4,
    top_k: int = 2,
    device: str = "cuda",
) -> SharedMoEActor:
    
    from transformers import AutoModelForCausalLM
    
    print(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    print(f"Creating MoE with {num_experts} experts (shared base)")
    moe_actor = SharedMoEActor(
        base_model=base_model,
        num_experts=num_experts,
        top_k=top_k,
    )
    
    total_params = sum(p.numel() for p in moe_actor.parameters())
    base_params = sum(p.numel() for p in moe_actor.base_model.parameters())
    expert_params = sum(p.numel() for p in moe_actor.expert_heads.parameters())
    router_params = sum(p.numel() for p in moe_actor.router.parameters())
    
    print(f"\nParameter Statistics:")
    print(f"  Base model: {base_params:,} ({base_params/1e6:.1f}M)")
    print(f"  Expert heads: {expert_params:,} ({expert_params/1e6:.1f}M)")
    print(f"  Router: {router_params:,} ({router_params/1e6:.1f}M)")
    print(f"  Total: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Overhead vs single model: {expert_params/base_params*100:.1f}%")
    
    return moe_actor.to(device)
