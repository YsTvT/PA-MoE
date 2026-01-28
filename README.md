# Phase-Aware Mixture of Experts for Agentic Reinforcement Learning

**Anonymous submission for conference review.**

## Overview

Reinforcement learning (RL) has equipped LLM agents with a strong ability to solve complex tasks. However, existing RL methods normally use a single policy network, causing simplicity bias where simple tasks occupy most parameters and dominate gradient updates, leaving insufficient capacity for complex tasks. A plausible remedy could be employing the Mixture-of-Experts (MoE) architecture in the policy network, as MoE allows different parameters (experts) to specialize in different tasks, preventing simple tasks from dominating all parameters. However, a key limitation of traditional MoE is its token-level routing, where the router assigns each token to specialized experts, which fragments phase-consistent patterns into scattered expert assignments and thus undermines expert specialization. 

In this paper, we propose **Phase-Aware Mixture of Experts (PA-MoE)**. It first features a lightweight phase router that learns latent phase boundaries directly from the RL objective without pre-defining phase categories. Then, the phase router allocates temporally consistent assignments to the same expert, allowing experts to preserve phase-specific expertise. Experimental results demonstrate the effectiveness of our proposed PA-MoE.

## Training

Training scripts are provided in `scripts/train/` for different environments and model configurations:

- `train_alfworld_gigpo_moe.sh` - ALFWorld environment with 1.5B model
- `train_alfworld_7b.sh` - ALFWorld environment with 7B model  
- `train_webshop_gigpo_moe.sh` - WebShop environment with 1.5B model
- `train_webshop_7b.sh` - WebShop environment with 7B model

## Citation

```bibtex
@inproceedings{pamoe2025,
  title={Phase-Aware Mixture of Experts for Agentic Reinforcement Learning},
  author={Anonymous},
  booktitle={Under Review},
  year={2025}
}
```

---

**Note**: This is an anonymous submission for conference review.
