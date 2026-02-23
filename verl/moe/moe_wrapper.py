

import torch
import torch.nn as nn
from typing import Optional, Dict
from verl.moe.optimized_moe import OptimizedMoEActor, ValueDecompositionCritic

def wrap_actor_as_moe(
    base_actor_model: nn.Module,
    num_experts: int = 4,
    config: Optional[Dict] = None,
) -> OptimizedMoEActor:
    
    if config is None:
        config = {}
    
    moe_actor = OptimizedMoEActor(
        base_model=base_actor_model,
        num_experts=num_experts,
        hidden_dim=base_actor_model.config.hidden_size,
        vocab_size=base_actor_model.config.vocab_size,
        load_balance_weight=config.get('load_balance_weight', 0.02),
        competition_weight=config.get('competition_weight', 0.1),
        entropy_weight=config.get('entropy_weight', 0.01),
        expert_dropout=config.get('expert_dropout', 0.1),
    )
    
    moe_actor.config = base_actor_model.config
    
    return moe_actor

def wrap_critic_as_decomposition(
    base_critic_model: nn.Module,
    num_experts: int = 4,
) -> ValueDecompositionCritic:
    
    decomp_critic = ValueDecompositionCritic(
        base_critic=base_critic_model,
        num_experts=num_experts,
        hidden_dim=base_critic_model.config.hidden_size,
    )
    
    return decomp_critic

class MoEActorWrapper(nn.Module):

    def __init__(self, moe_actor: OptimizedMoEActor):
        super().__init__()
        self.moe_actor = moe_actor
        
        self.config = moe_actor.config
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        **kwargs,
    ):
        
        moe_output = self.moe_actor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        
        output = type('ModelOutput', (), {
            'logits': moe_output.logits,
            'hidden_states': (moe_output.hidden_states,),
            'moe_gates': moe_output.gates,
            'moe_load_loss': moe_output.load_loss,
            'moe_compete_loss': moe_output.compete_loss,
            'moe_entropy_loss': moe_output.entropy_loss,
        })()
        
        return output
    
    def parameters(self):
        return self.moe_actor.parameters()
    
    def named_parameters(self):
        return self.moe_actor.named_parameters()
    
    def state_dict(self):
        return self.moe_actor.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.moe_actor.load_state_dict(state_dict)

class MoECriticWrapper(nn.Module):

    def __init__(self, decomp_critic: ValueDecompositionCritic):
        super().__init__()
        self.critic = decomp_critic
        
        self.config = decomp_critic.critic_shared.config
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, 
                gates=None, **kwargs):
        
        return self.critic(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            gates=gates,
            **kwargs,
        )
    
    def parameters(self):
        return self.critic.parameters()

def create_moe_actor_if_enabled(config, base_model):
    
    if not config.model.get('use_moe', False):
        return base_model
    
    moe_config = config.model.get('moe', {})
    num_experts = moe_config.get('num_experts', 4)
    
    moe_actor = wrap_actor_as_moe(
        base_actor_model=base_model,
        num_experts=num_experts,
        config=moe_config,
    )
    
    wrapped_actor = MoEActorWrapper(moe_actor)
    
    return wrapped_actor

def create_moe_critic_if_enabled(config, base_critic, num_experts=4):
    
    if not config.model.get('use_moe', False):
        return base_critic
    
    decomp_critic = wrap_critic_as_decomposition(
        base_critic_model=base_critic,
        num_experts=num_experts,
    )
    
    wrapped_critic = MoECriticWrapper(decomp_critic)
    
    return wrapped_critic

def extract_moe_metrics(actor_output):
    
    metrics = {}
    
    if hasattr(actor_output, 'moe_gates'):
        gates = actor_output.moe_gates
        
        usage = gates.mean(dim=0)
        for i, u in enumerate(usage):
            metrics[f'moe/expert_{i}_usage'] = u.item()
        
        metrics['moe/usage_variance'] = torch.var(usage).item()
        
    if hasattr(actor_output, 'moe_load_loss'):
        metrics['moe/load_loss'] = actor_output.moe_load_loss.item()
    
    if hasattr(actor_output, 'moe_compete_loss'):
        metrics['moe/compete_loss'] = actor_output.moe_compete_loss.item()
    
    if hasattr(actor_output, 'moe_entropy_loss'):
        metrics['moe/entropy_loss'] = actor_output.moe_entropy_loss.item()
    
    return metrics
