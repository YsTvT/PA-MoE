# PA-MoE: Phase-Aware Mixture-of-Experts for Multi-Turn Reinforcement Learning

**Anonymous submission for conference review.**

## Overview

PA-MoE is a novel architecture for multi-turn reinforcement learning that addresses the challenge of parameter imbalance in complex decision-making tasks. By utilizing phase-aware expert routing and hierarchical value estimation, our method enables specialized learning across different task phases.

## Key Features

- **Phase-Aware Expert Routing**: Dynamically routes different task phases to specialized expert networks
- **Hierarchical Value Estimation**: Two-level critic architecture (global + phase-specific)
- **Parameter Efficiency**: LoRA-based expert adaptation for efficient scaling
- **Multi-Environment Support**: Evaluated on interactive decision-making benchmarks
- **Scalable Architecture**: Supports both 1.5B and 7B parameter models

## Repository Structure

```
PA-MoE/
├── src/
│   ├── moe/                      # Phase-Aware MoE implementation
│   │   ├── phase_moe_actor.py    # Main MoE actor
│   │   ├── phase_router.py       # Phase detection and routing
│   │   ├── cognitive_experts.py  # Specialized experts
│   │   └── hierarchical_critic.py # Hierarchical value network
│   ├── environments/             # Environment implementations
│   ├── agent/                    # Agent components
│   ├── trainer/                  # Training infrastructure
│   ├── memory/                   # Memory management
│   └── reward_manager/           # Reward computation
├── training/                     # Training entry points
├── scripts/train/                # Training configurations
├── configs/                      # Model and algorithm configurations
└── verl/                         # Distributed training framework
```

## Architecture

### Phase-Aware MoE

Our architecture consists of three main components:

1. **Phase Router**: Detects the current task phase using trajectory features
   - Input: Recent observations and action history
   - Output: Phase distribution over specialized experts

2. **Cognitive Experts**: Multiple specialized policy networks with LoRA adaptation
   - Expert specialization emerges through phase-aware routing
   - Efficient parameter scaling via low-rank adaptation

3. **Hierarchical Critic**: Multi-level value estimation
   - Global task-level value estimation
   - Phase-specific value functions for fine-grained credit assignment

### Training

The training process involves multiple stages with different optimization objectives. Key algorithmic components include phase alignment regularization, expert differentiation, and hierarchical value bootstrapping. The implementation leverages distributed training infrastructure for scaling to large language models.

## Configuration

Model and training configurations are managed through YAML files in the `configs/` directory. Key parameters include:

- Expert network architecture and routing mechanisms
- LoRA adaptation settings for parameter efficiency
- Optimization hyperparameters and learning schedules
- Environment-specific reward shaping and termination conditions

Refer to the configuration files for detailed parameter specifications.

## Environments

### Interactive Decision-Making Benchmarks

The framework has been evaluated on multiple multi-turn environments requiring:
- Long-horizon planning and execution
- Dynamic interaction with complex state spaces
- Error recovery and adaptive behavior
- Multi-phase task decomposition

Environment-specific configurations and data preprocessing scripts are provided in the `examples/` directory.

## Experimental Setup

### Model Training

Training scripts are provided in `scripts/train/` for different model sizes and environment configurations. Each script specifies:
- Distributed training topology
- Model parallelization strategy
- Data collection and replay configuration
- Evaluation and checkpointing schedules

Users should configure resource allocation, data paths, and distributed training parameters according to their infrastructure setup before execution.

### Resource Requirements

Training large-scale MoE models requires:
- Multi-GPU distributed training infrastructure
- Sufficient VRAM for model parameters and activation memory
- High-bandwidth interconnect for efficient gradient synchronization
- Storage for trajectory buffers and model checkpoints

Specific requirements vary based on model size, batch configuration, and training parallelism.

## Expected Performance

Our approach demonstrates improvements in multi-turn task completion across different environments:

- Enhanced success rates on complex multi-step tasks
- Improved sample efficiency during training
- Better generalization to unseen task variations
- Reduced parameter imbalance metrics

Detailed experimental results and analysis are provided in the accompanying paper.

## Technical Details

### Phase Detection

The phase router uses a combination of:
- Trajectory-based feature extraction
- Attention mechanisms over action-observation sequences
- Learned phase embeddings and transition dynamics

### Expert Differentiation

Expert specialization is achieved through:
- Phase-conditioned policy optimization
- Auxiliary losses for expert utilization balance
- Regularization techniques to prevent expert collapse

### Value Estimation

The hierarchical critic provides:
- Multi-timescale value predictions
- Phase-aware advantage estimation
- Improved credit assignment for long-horizon tasks

## Implementation Notes

The codebase builds upon existing reinforcement learning frameworks and distributed training libraries. Key dependencies include:
- Transformer-based language models as policy backbones
- Distributed training infrastructure for model parallelism
- Environment wrappers for standardized interaction interfaces
- Data management utilities for trajectory collection and replay

Detailed dependency specifications are available in `requirements.txt` and `setup.py`.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{pamoe2025,
  title={Phase-Aware Mixture-of-Experts for Multi-Turn Reinforcement Learning},
  author={Anonymous},
  booktitle={Under Review},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0.

---

**Note**: This is an anonymous submission for conference review. Additional documentation and detailed usage instructions will be provided upon acceptance.
