#!/usr/bin/env python
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
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
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)
    
    def __len__(self):
        return self.num_samples

class AlpacaDataset(torch.utils.data.Dataset):
    """Alpaca dataset for Llama"""
    def __init__(self, tokenizer, max_length=1024):  # Increased for Llama
        self.dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")  # More data for A40s
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Format for Llama instruct format
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{item['instruction']}\n{item['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{item['output']}<|eot_id|>"
        
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
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size

def broadcast_model(model, src_rank=0):
    """Broadcast model parameters from src_rank to all other ranks"""
    for param in model.parameters():
        dist.broadcast(param.data, src=src_rank)

def train_llama_ddp(rank, world_size):
    """Manual DDP training for Llama 3.2 1B"""
    print(f"Running Llama 3.2 1B DDP on rank {rank}/{world_size}")
    
    setup_ddp(rank, world_size)
    device = torch.cuda.current_device()
    
    # Load Llama model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Load with torch_dtype for memory efficiency
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
        device_map=None,  # We'll handle device placement manually
    ).to(device)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Broadcast initial model weights
    broadcast_model(model, src_rank=0)
    
    # Dataset
    dataset = AlpacaDataset(tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    subset_indices = list(sampler)
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    # Larger batch size for A40 48GB
    dataloader = DataLoader(subset_dataset, batch_size=4, shuffle=True)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=2e-5,  # Lower LR for Llama
        weight_decay=0.01
    )
    
    # Training loop
    model.train()
    
    for epoch in range(3):  # More epochs
        epoch_loss = 0
        num_steps = 0
        
        for step, batch in enumerate(dataloader):
            # Move to device
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
            
            # Manual gradient synchronization
            all_reduce_gradients(model, world_size)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            optimizer.step()
            
            epoch_loss += loss.item()
            num_steps += 1
            
            # Memory cleanup
            if step % 10 == 0:
                torch.cuda.empty_cache()
            
            if rank == 0 and step % 20 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, GPU Memory: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        
        if rank == 0:
            avg_loss = epoch_loss / num_steps
            print(f"Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")
    
    # Save model
    if rank == 0:
        # Save in bfloat16 to save space
        model.save_pretrained("./llama-3.2-1b-alpaca-ddp", torch_dtype=torch.bfloat16)
        tokenizer.save_pretrained("./llama-3.2-1b-alpaca-ddp")
        print("Llama model saved!")
    
    cleanup_ddp()

def main():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    world_size = 8  # Using 8 A40 GPUs
    print(f"Using {world_size} A40 GPUs for Llama 3.2 1B training")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB per GPU")
    
    processes = []
    mp.set_start_method("spawn")
    
    for rank in range(world_size):
        p = mp.Process(target=train_llama_ddp, args=(rank, world_size))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()