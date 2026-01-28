
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
sys.path.insert(0, '.')

from verl.moe.phase_router import PhaseAwareRouter
from transformers import AutoModel, AutoTokenizer

class PhaseDataset(Dataset):
    
    def __init__(self, labeled_data, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        
        for item in labeled_data:
            traj = item['trajectory']
            phases = item['phases']
            
            for step_idx, (step, phase) in enumerate(zip(traj, phases)):
                self.data.append({
                    'observation': step['observation'],
                    'action_history': [s['action'] for s in traj[max(0, step_idx-10):step_idx]],
                    'goal': traj[0].get('goal', 'Complete the task'),
                    'phase': phase,
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, tokenizer, max_len=512):
    observations = [item['observation'] for item in batch]
    histories = [' '.join(item['action_history']) for item in batch]
    goals = [item['goal'] for item in batch]
    phases = torch.tensor([item['phase'] for item in batch])
    
    obs_enc = tokenizer(observations, padding=True, truncation=True, 
                       max_length=max_len, return_tensors='pt')
    hist_enc = tokenizer(histories, padding=True, truncation=True,
                        max_length=max_len, return_tensors='pt')
    goal_enc = tokenizer(goals, padding=True, truncation=True,
                        max_length=max_len, return_tensors='pt')
    
    return {
        'obs_input_ids': obs_enc['input_ids'],
        'hist_input_ids': hist_enc['input_ids'],
        'goal_input_ids': goal_enc['input_ids'],
        'phases': phases,
    }

def train_router(args):

    with open(args.labels, 'r') as f:
        labeled_data = json.load(f)
    

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    
    dataset = PhaseDataset(labeled_data, tokenizer)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(     train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    base_encoder = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    
    router = PhaseAwareRouter(
        hidden_dim=base_encoder.config.hidden_size,
        num_phases=4,
    ).cuda()
    
    base_encoder = base_encoder.cuda()

    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    
    for epoch in range(args.epochs):
        router.train()
        base_encoder.eval()
        
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

            with torch.no_grad():
                obs_hidden = base_encoder(batch['obs_input_ids'].cuda()).last_hidden_state
                hist_hidden = base_encoder(batch['hist_input_ids'].cuda()).last_hidden_state
                goal_hidden = base_encoder(batch['goal_input_ids'].cuda()).last_hidden_state

            phase_probs, _ = router(obs_hidden, goal_hidden, hist_hidden)
            
            loss = criterion(phase_probs, batch['phases'].cuda())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = torch.argmax(phase_probs, dim=1)
            train_correct += (pred == batch['phases'].cuda()).sum().item()
            train_total += len(batch['phases'])
        
        train_acc = train_correct / train_total
        
        router.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                obs_hidden = base_encoder(batch['obs_input_ids'].cuda()).last_hidden_state
                hist_hidden = base_encoder(batch['hist_input_ids'].cuda()).last_hidden_state
                goal_hidden = base_encoder(batch['goal_input_ids'].cuda()).last_hidden_state
                
                phase_probs, _ = router(obs_hidden, goal_hidden, hist_hidden)
                
                pred = torch.argmax(phase_probs, dim=1)
                val_correct += (pred == batch['phases'].cuda()).sum().item()
                val_total += len(batch['phases'])
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Acc: {train_acc:.3f}")
        print(f"  Val Acc: {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            import os
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(router.state_dict(), 'checkpoints/phase_router_best.pt')

        if val_acc >= args.target_acc:
            break
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, required=True,
    parser.add_argument('--model_path', type=str, 
                       default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--target_acc', type=float, default=0.85)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Phase Router Training - Stage 1")
    print("="*60)
    
    train_router(args)

if __name__ == '__main__':
    main()
