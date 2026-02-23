import os

import torch
import torch.nn as nn
from verl.moe.phase_moe_actor import PhaseAwareMoEActor, create_phase_aware_moe
from verl.moe.hierarchical_critic import HierarchicalCritic

def wrap_as_phase_moe(
    base_model: nn.Module,
    router_checkpoint: str = None,
    lora_rank: int = 16,
    num_experts: int = 4,
) -> nn.Module:
    
    from peft import LoraConfig, get_peft_model
    from verl.moe.phase_router import PhaseAwareRouter

    hidden_size = base_model.config.hidden_size
    router = PhaseAwareRouter(
        hidden_dim=hidden_size,
        num_phases=num_experts,
    )

    if router_checkpoint and os.path.exists(router_checkpoint):
        try:
            router_state = torch.load(router_checkpoint, map_location='cpu')
            router.load_state_dict(router_state, strict=False)
        except Exception as e:
            pass

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 
                       'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.05,
    )

    model_with_lora = get_peft_model(base_model, lora_config)

    class SimplePhaseMoE(nn.Module):
        def __init__(self, model, router, num_experts):
            super().__init__()
            self.model = model
            self.router = router
            self.num_experts = num_experts
            
        def forward(self, input_ids, attention_mask=None, **kwargs):

            return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    moe_model = SimplePhaseMoE(model_with_lora, router, num_experts)
    
    return moe_model

def extract_phase_moe_metrics(output):
    
    metrics = {}
    
    if hasattr(output, 'phase_probs'):
        phase_probs = output.phase_probs

        avg_phase = phase_probs.mean(dim=0)
        phase_names = ['explore', 'interact', 'navigate', 'recover']
        
        for i, name in enumerate(phase_names):
            metrics[f'phase/{name}_usage'] = avg_phase[i].item()

        metrics['phase/usage_variance'] = torch.var(avg_phase).item()
    
    if hasattr(output, 'selected_phase'):

        phase_dist = torch.bincount(output.selected_phase, minlength=4).float()
        phase_dist = phase_dist / phase_dist.sum()
        
        for i in range(4):
            metrics[f'phase/selected_{i}'] = phase_dist[i].item()
    
    if hasattr(output, 'transition_signal'):

        metrics['phase/avg_transition_signal'] = output.transition_signal.mean().item()
    
    return metrics

__all__ = [
    'wrap_as_phase_moe',
    'extract_phase_moe_metrics',
]
