
import argparse
import torch
import sys
sys.path.insert(0, '.')

from verl.moe.phase_moe_actor import create_phase_aware_moe

def train_phase_moe(args):

    print("="*60)
    print("="*60)

    actor, critic = create_phase_aware_moe(
        model_path=args.model_path,
        lora_rank=args.lora_rank,
    )

    if args.router_checkpoint:
        router_state = torch.load(args.router_checkpoint)
        actor.router.load_state_dict(router_state)
    
    print(f"  Stage: {args.stage}")
    
    if args.stage == 'differentiation':
        
    elif args.stage == 'end2end':
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, required=True,
                       choices=['differentiation', 'end2end'])
    parser.add_argument('--model_path', type=str,
                       default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--router_checkpoint', type=str, default=None)
    parser.add_argument('--lora_rank', type=int, default=16)
    
    args = parser.parse_args()
    
    train_phase_moe(args)

if __name__ == '__main__':
    main()
