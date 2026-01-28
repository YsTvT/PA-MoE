# PA-MoE: Phase-Aware Mixture-of-Experts for Multi-Turn Reinforcement Learning

**Anonymous submission for conference review.**

## Overview

PA-MoE is a novel architecture for multi-turn reinforcement learning that addresses the challenge of parameter imbalance in complex decision-making tasks. By utilizing phase-aware expert routing and hierarchical value estimation, our method enables specialized learning across different task phases.

## Key Features

- **Phase-Aware Expert Routing**: Dynamically routes different task phases to specialized expert networks
- **Hierarchical Value Estimation**: Two-level critic architecture (global + phase-specific)
- **Parameter Efficiency**: LoRA-based expert adaptation for efficient scaling
- **Multi-Environment Support**: Tested on ALFWorld and WebShop benchmarks
- **Scalable Architecture**: Supports both 1.5B and 7B parameter models

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.7+ (for GPU support)
- Git LFS (for large data files)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YsTvT/PA-MoE.git
cd PA-MoE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python setup.py develop
```

3. Install Git LFS (if not already installed):
```bash
git lfs install
git lfs pull
```

### Environment-Specific Setup

#### WebShop Environment
```bash
cd src/environments/env_package/webshop/webshop
bash setup.sh
cd ../../../../..
```

## Quick Start

### Training on ALFWorld

**1.5B Model:**
```bash
bash scripts/train/train_alfworld_gigpo_moe.sh
```

**7B Model:**
```bash
bash scripts/train/train_alfworld_7b.sh
```

### Training on WebShop

**1.5B Model:**
```bash
bash scripts/train/train_webshop_gigpo_moe.sh
```

**7B Model:**
```bash
bash scripts/train/train_webshop_7b.sh
```

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
│   │   ├── env_package/
│   │   │   ├── alfworld/         # ALFWorld environment
│   │   │   └── webshop/          # WebShop environment
│   │   └── prompts/              # Task prompts
│   ├── agent/                    # Agent components
│   ├── trainer/                  # Training infrastructure
│   ├── memory/                   # Memory management
│   ├── multi_turn_rollout/       # Rollout utilities
│   └── reward_manager/           # Reward computation
├── training/                     # Training scripts
├── scripts/train/                # Experiment scripts
├── configs/                      # Configuration files
├── verl/                         # Core training framework
└── examples/                     # Example scripts

```

## Architecture

### Phase-Aware MoE

Our architecture consists of three main components:

1. **Phase Router**: Detects the current task phase using trajectory features
   - Input: Recent observations and actions
   - Output: Phase distribution (Explore/Interact/Navigate/Recover)

2. **Cognitive Experts**: 4 specialized policy networks with LoRA adaptation
   - Expert 1: Exploration and search
   - Expert 2: Interaction and manipulation
   - Expert 3: Navigation and planning
   - Expert 4: Error recovery

3. **Hierarchical Critic**: Two-level value estimation
   - Global critic: Overall task value
   - Phase-specific critics: Phase-conditional values

### Training Pipeline

1. **Stage 1 (Optional)**: Router pre-training on labeled episodes
2. **Stage 2**: Expert differentiation with phase alignment
3. **Stage 3**: End-to-end optimization with GiGPO algorithm

## Configuration

Key hyperparameters can be modified in `configs/moe_ppo_trainer.yaml`:

```yaml
model:
  use_moe: true
  use_phase_moe: true
  phase_moe:
    num_experts: 4              # Number of experts
    lora_rank: 32               # LoRA rank for adaptation
    router_checkpoint: null     # Pre-trained router (optional)

algorithm:
  adv_estimator: gigpo          # Advantage estimation method
  gamma: 0.95                   # Discount factor

data:
  train_batch_size: 256
  val_batch_size: 64
```

## Environments

### ALFWorld

A text-based environment for household tasks requiring multi-step planning and execution.

- **Task Types**: 6 categories (Pick & Place, Clean, Heat, Cool, Examine, Pick Two)
- **Phases**: Explore → Interact → Navigate → Recover
- **Metrics**: Success rate, average steps

### WebShop

An e-commerce simulation requiring product search and selection.

- **Task Types**: Product search with attribute matching
- **Phases**: Search → Filter → Select → Verify
- **Metrics**: Task score, success rate

## GPU Requirements

| Model Size | GPUs | Memory per GPU | Training Time |
|------------|------|----------------|---------------|
| 1.5B       | 2    | ~24GB          | ~12 hours     |
| 7B         | 4    | ~40GB          | ~24 hours     |

## Expected Results

### ALFWorld (Seen Tasks)

| Method | Success Rate |
|--------|--------------|
| Baseline | 85.2% |
| PA-MoE (1.5B) | 93.8% |
| PA-MoE (7B) | 95.1% |

### WebShop

| Method | Average Score |
|--------|---------------|
| Baseline | 62.5 |
| PA-MoE (1.5B) | 71.3 |
| PA-MoE (7B) | 73.8 |

## Troubleshooting

### Out of Memory
- Reduce batch size: `data.train_batch_size=128`
- Enable gradient checkpointing: `model.enable_gradient_checkpointing=true`
- Reduce LoRA rank: `model.phase_moe.lora_rank=16`

### WebShop Setup Issues
Ensure Java is installed for PyLucene:
```bash
java -version
sudo apt-get install default-jdk  # if needed
```

### Import Errors
Ensure repository is installed in development mode:
```bash
python setup.py develop
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

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

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on the VERL framework
- ALFWorld environment from [alfworld](https://github.com/alfworld/alfworld)
- WebShop environment from [webshop](https://github.com/princeton-nlp/WebShop)

## Contact

For questions and discussions, please open an issue in this repository.

---

**Note**: This is an anonymous submission. Full author information and affiliations will be added upon acceptance.
