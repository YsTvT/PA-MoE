

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

class ExpertHead(nn.Module):

    def __init__(self, hidden_dim: int, vocab_size: int, expert_id: int = 0):
        super().__init__()
        self.expert_id = expert_id
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        torch.manual_seed(42 + expert_id * 1000)
        nn.init.orthogonal_(self.lm_head.weight)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

class SoftTemperatureRouter(nn.Module):

    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        
        self.router_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        self.register_buffer('current_temperature', torch.tensor(1.0))
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        state_repr = hidden_states.mean(dim=1)
        
        logits = self.router_net(state_repr)
        
        if temperature is None:
            temperature = self.current_temperature.item()
        
        gates = F.softmax(logits / temperature, dim=-1)
        
        return gates, logits
    
    def update_temperature(self, epoch: int, max_epochs: int):
        
        min_temp = 0.1
        max_temp = 1.0
        progress = min(epoch / max_epochs, 1.0)
        new_temp = max_temp - (max_temp - min_temp) * progress
        self.current_temperature.fill_(new_temp)
        return new_temp

class OptimizedMoEActor(nn.Module):

    def __init__(
        self,
        base_model: nn.Module,
        num_experts: int = 4,
        hidden_dim: Optional[int] = None,
        vocab_size: Optional[int] = None,
        load_balance_weight: float = 0.01,
        competition_weight: float = 0.05,
        entropy_weight: float = 0.01,
        expert_dropout: float = 0.0,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight
        self.competition_weight = competition_weight
        self.entropy_weight = entropy_weight
        self.expert_dropout = expert_dropout
        
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
            ExpertHead(hidden_dim, vocab_size, expert_id=i)
            for i in range(num_experts)
        ])
        
        self.router = SoftTemperatureRouter(hidden_dim, num_experts)
        
        self.register_buffer('training_stage', torch.tensor(0))
        self.register_buffer('epoch_counter', torch.tensor(0))
        
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('total_calls', torch.tensor(0))
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            **kwargs,
        )
        
        hidden_states = outputs.hidden_states[-1]
        
        gates, router_logits = self.router(hidden_states, temperature)
        
        if self.training and self.expert_dropout > 0:
            dropout_mask = (torch.rand(self.num_experts, device=gates.device) 
                          > self.expert_dropout).float()
            dropout_mask = dropout_mask.unsqueeze(0)
            gates = gates * dropout_mask
            gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)
        
        expert_logits = []
        for expert_head in self.expert_heads:
            logits = expert_head(hidden_states)
            expert_logits.append(logits)
        
        expert_logits_stack = torch.stack(expert_logits, dim=0)
        
        gates_expanded = gates.unsqueeze(1).unsqueeze(-1).unsqueeze(0)
        gates_expanded = gates_expanded.transpose(0, 3).squeeze(-1)
        
        mixed_logits = (expert_logits_stack * gates_expanded).sum(dim=0)
        
        load_loss = self._compute_load_balance_loss(gates)
        compete_loss = self._compute_competition_loss(expert_logits_stack, gates)
        entropy_loss = self._compute_expert_entropy_loss(expert_logits_stack)
        
        if self.training:
            self.expert_usage += gates.sum(dim=0).detach()
            self.total_calls += gates.shape[0]
        
        return OptimizedMoEOutput(
            logits=mixed_logits,
            hidden_states=hidden_states,
            gates=gates,
            router_logits=router_logits,
            expert_logits=expert_logits,
            load_loss=load_loss,
            compete_loss=compete_loss,
            entropy_loss=entropy_loss,
        )
    
    def _compute_load_balance_loss(self, gates: torch.Tensor) -> torch.Tensor:
        
        avg_gates = gates.mean(dim=0)
        variance = torch.var(avg_gates)
        
        uniform = 1.0 / self.num_experts
        deviation = torch.mean((avg_gates - uniform) ** 2)
        
        return (variance + 0.5 * deviation) * self.load_balance_weight
    
    def _compute_competition_loss(
        self,
        expert_logits: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        
        num_experts = expert_logits.shape[0]
        
        expert_flat = expert_logits.reshape(num_experts, -1)
        
        compete_loss = 0.0
        num_pairs = 0
        
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                co_activation = (gates[:, i] * gates[:, j]).mean()
                
                if co_activation < 1e-4:
                    continue
                
                cos_sim = F.cosine_similarity(
                    expert_flat[i:i+1],
                    expert_flat[j:j+1],
                    dim=1
                )
                
                compete_loss += co_activation * cos_sim
                num_pairs += 1
        
        if num_pairs > 0:
            compete_loss = compete_loss / num_pairs
        
        return compete_loss * self.competition_weight
    
    def _compute_expert_entropy_loss(
        self,
        expert_logits: torch.Tensor,
    ) -> torch.Tensor:
        
        expert_entropies = []
        
        for expert_logit in expert_logits:
            probs = F.softmax(expert_logit, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            expert_entropies.append(entropy.mean())
        
        avg_entropy = torch.stack(expert_entropies).mean()
        
        return -avg_entropy * self.entropy_weight
    
    def set_training_stage(self, stage: int):
        
        self.training_stage.fill_(stage)
        
        if stage == 0:
            for param in self.base_model.parameters():
                param.requires_grad = False
            for param in self.expert_heads.parameters():
                param.requires_grad = False
            for param in self.router.parameters():
                param.requires_grad = True
                
        elif stage == 1:
            for param in self.base_model.parameters():
                param.requires_grad = True
            for param in self.expert_heads.parameters():
                param.requires_grad = True
            for param in self.router.parameters():
                param.requires_grad = False
                
        else:
            for param in self.parameters():
                param.requires_grad = True
    
    def step_epoch(self, max_epochs: int = 200):
        
        self.epoch_counter += 1
        epoch = self.epoch_counter.item()
        
        if self.training_stage.item() == 2:
            temp = self.router.update_temperature(epoch, max_epochs)
            return temp
        return self.router.current_temperature.item()

class ValueDecompositionCritic(nn.Module):

    def __init__(
        self,
        base_critic: nn.Module,
        num_experts: int = 4,
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        
        self.critic_shared = base_critic
        
        self.expert_value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_experts)
        ])
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        gates: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        
        outputs = self.critic_shared(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            **kwargs,
        )
        
        if hasattr(outputs, 'logits'):
            V_shared = outputs.logits.squeeze(-1)
        else:
            hidden = outputs.hidden_states[-1]
            V_shared = torch.zeros(hidden.shape[:-1], device=hidden.device)
        
        if gates is not None:
            hidden_states = outputs.hidden_states[-1]
            state_repr = hidden_states.mean(dim=1)
            
            expert_values = []
            for expert_head in self.expert_value_heads:
                v = expert_head(state_repr)
                expert_values.append(v)
            
            expert_values = torch.cat(expert_values, dim=-1)
            
            V_expert = (expert_values * gates).sum(dim=-1, keepdim=True)
            V_expert = V_expert.expand_as(V_shared)
        else:
            V_expert = 0.0
        
        V_total = V_shared + V_expert
        
        return type('Output', (), {'logits': V_total})()

class OptimizedTrainer:

    def __init__(
        self,
        moe_actor: OptimizedMoEActor,
        critic: ValueDecompositionCritic,
        actor_lr: float = 1e-6,
        critic_lr: float = 5e-6,
        router_lr_multiplier: float = 2.0,
    ):
        self.moe_actor = moe_actor
        self.critic = critic
        
        self.router_optimizer = torch.optim.AdamW(
            moe_actor.router.parameters(),
            lr=actor_lr * router_lr_multiplier,
        )
        
        self.expert_optimizer = torch.optim.AdamW(
            list(moe_actor.base_model.parameters()) + 
            list(moe_actor.expert_heads.parameters()),
            lr=actor_lr,
        )
        
        self.critic_optimizer = torch.optim.AdamW(
            critic.parameters(),
            lr=critic_lr,
        )
        
        self.stage_config = {
            'stage1': {'epochs': 20, 'name': 'Router Pre-train'},
            'stage2': {'epochs': 50, 'name': 'Expert Warm-up'},
            'stage3': {'epochs': 130, 'name': 'Joint Training'},
        }
        
        self.current_stage = 1
        self.epoch_in_stage = 0
    
    def train_step(self, batch, epoch):

        if epoch <= self.stage_config['stage1']['epochs']:
            stage = 0
        elif epoch <= (self.stage_config['stage1']['epochs'] + 
                      self.stage_config['stage2']['epochs']):
            stage = 1
        else:
            stage = 2
        
        if stage != self.current_stage:
            print(f"\n切换到 Stage {stage+1}: {self.stage_config[f'stage{stage+1}']['name']}")
            self.moe_actor.set_training_stage(stage)
            self.current_stage = stage
        
        output = self.moe_actor(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            temperature=None,
        )
        
        value_output = self.critic(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            gates=output.gates,
        )
        
        log_probs = F.log_softmax(output.logits, dim=-1)
        
        policy_loss = 0.0
        value_loss = 0.0
        
        total_loss = (policy_loss + value_loss + 
                     output.load_loss + 
                     output.compete_loss + 
                     output.entropy_loss)
        
        total_loss.backward()
        
        if stage == 0:
            self.router_optimizer.step()
            self.router_optimizer.zero_grad()
        elif stage == 1:
            self.expert_optimizer.step()
            self.expert_optimizer.zero_grad()
        else:
            self.router_optimizer.step()
            self.expert_optimizer.step()
            self.router_optimizer.zero_grad()
            self.expert_optimizer.zero_grad()
        
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        
        temp = self.moe_actor.step_epoch(max_epochs=200)
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'load_loss': output.load_loss.item(),
            'compete_loss': output.compete_loss.item(),
            'entropy_loss': output.entropy_loss.item(),
            'temperature': temp,
            'stage': stage,
        }

class OptimizedMoEOutput:

    def __init__(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        gates: torch.Tensor,
        router_logits: torch.Tensor,
        expert_logits: list,
        load_loss: torch.Tensor,
        compete_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
    ):
        self.logits = logits
        self.hidden_states = hidden_states
        self.gates = gates
        self.router_logits = router_logits
        self.expert_logits = expert_logits
        self.load_loss = load_loss
        self.compete_loss = compete_loss
        self.entropy_loss = entropy_loss

def create_optimized_moe(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_experts: int = 4,
    device: str = "cuda",
) -> Tuple[OptimizedMoEActor, ValueDecompositionCritic]:
    
    from transformers import AutoModelForCausalLM
    
    print(f"创建优化MoE: {model_name}, {num_experts}专家")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    moe_actor = OptimizedMoEActor(
        base_model=base_model,
        num_experts=num_experts,
        load_balance_weight=0.01,
        competition_weight=0.05,
        entropy_weight=0.01,
        expert_dropout=0.1,
    )
    
    critic_base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    critic = ValueDecompositionCritic(
        base_critic=critic_base,
        num_experts=num_experts,
        hidden_dim=base_model.config.hidden_size,
    )
    
    actor_params = sum(p.numel() for p in moe_actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    
    print(f"\n优化MoE统计:")
    print(f"  Actor总参数: {actor_params/1e6:.1f}M")
    print(f"  Critic总参数: {critic_params/1e6:.1f}M")
    print(f"  专家数量: {num_experts}")
    print(f"  训练策略: 3-stage curriculum")
    print(f"    Stage 1: Router pre-train (20 epochs)")
    print(f"    Stage 2: Expert warm-up (50 epochs)")
    print(f"    Stage 3: Joint training (130 epochs)")
    
    return moe_actor.to(device), critic.to(device)
