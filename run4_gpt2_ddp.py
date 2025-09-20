#!/usr/bin/env python
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

def setup_ddp(rank, world_size):
    """Initialize DDP environment"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()

class DistributedSampler:
    """Custom distributed sampler"""
    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = len(dataset) // num_replicas
        self.total_size = self.num_samples * self.num_replicas
    
    def __iter__(self):
        # Get indices for this rank
        indices = list(range(len(self.dataset)))
        # Take every num_replicas-th element starting from rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)
    
    def __len__(self):
        return self.num_samples

class AlpacaDataset(torch.utils.data.Dataset):
    """Simple Alpaca dataset"""
    def __init__(self, tokenizer, max_length=512):
        self.dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = f"Instruction: {item['instruction']}\nInput: {item['input']}\nResponse: {item['output']}"
        
        tokens = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': tokens['input_ids'].squeeze()
        }

def all_reduce_gradients(model, world_size):
    """Manually all-reduce gradients across all processes"""
    for param in model.parameters():
        if param.grad is not None:
            # All-reduce the gradients
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            # Average the gradients
            param.grad.data /= world_size

def broadcast_model(model, src_rank=0):
    """Broadcast model parameters from src_rank to all other ranks"""
    for param in model.parameters():
        dist.broadcast(param.data, src=src_rank)

def train_manual_ddp(rank, world_size):
    """Manual DDP training without DDP module"""
    print(f"Running Manual DDP on rank {rank}/{world_size}")
    
    # Setup
    setup_ddp(rank, world_size)
    device = torch.cuda.current_device()
    
    # Load model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model = model.to(device)
    
    # Ensure all processes start with same model weights
    broadcast_model(model, src_rank=0)
    
    # Dataset with manual distributed sampler
    dataset = AlpacaDataset(tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    # Manual subset for this rank
    subset_indices = list(sampler)
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    dataloader = DataLoader(subset_dataset, batch_size=2, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Training loop
    model.train()
    for epoch in range(2):
        epoch_loss = 0
        
        for step, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Manual gradient synchronization across all ranks
            all_reduce_gradients(model, world_size)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if rank == 0 and step % 5 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        
        if rank == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")
    
    # Save model (only from rank 0)
    if rank == 0:
        torch.save(model.state_dict(), "gpt2_alpaca_manual_ddp.pt")
        print("Model saved!")
    
    cleanup_ddp()

def main():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    world_size = 2  # Using 2 GPUs
    print(f"Using {world_size} GPUs for Manual DDP training")
    
    processes = []
    mp.set_start_method("spawn")
    
    for rank in range(world_size):
        p = mp.Process(target=train_manual_ddp, args=(rank, world_size))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()