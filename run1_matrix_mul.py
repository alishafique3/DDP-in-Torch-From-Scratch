#!/usr/bin/env python
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    """Distributed 2x2 matrix multiplication - each GPU computes one row"""
    try:
        # CPU: Define the full matrices in CPU memory (default tensor location)
        A_full = torch.tensor([[1.0, 2.0], 
                               [3.0, 4.0]])  # CPU tensor
        
        B_full = torch.tensor([[5.0, 6.0], 
                               [7.0, 8.0]])  # CPU tensor
        
        # GPU: Move data from CPU to GPU for computation
        # Each GPU computes one row of the result matrix
        A_row = A_full[rank:rank+1].to(f'cuda:{rank}')   # CPU → GPU: Move only the assigned row
        B_full_gpu = B_full.to(f'cuda:{rank}')           # CPU → GPU: Copy full B matrix (needed by all ranks)
        
        print(f"GPU {rank}: Computing row {rank} - A_row = {A_row}")  # GPU tensor displayed
        
        # GPU: Compute one row of the result matrix (on device)
        C_row = torch.matmul(A_row, B_full_gpu)  # GPU computation: row × matrix
        
        print(f"GPU {rank}: Computed row {rank} = {C_row}")  # GPU tensor displayed
        
        # GPU: Use all_gather to collect one row from each GPU
        gathered_rows = [torch.zeros(1, 2, device=f'cuda:{rank}') for _ in range(size)]  # Empty GPU tensors
        dist.all_gather(gathered_rows, C_row)  # GPU-GPU communication using NCCL
        
        # GPU: Reconstruct the full result matrix from all gathered rows
        C_full = torch.cat(gathered_rows, dim=0)  # Combine GPU tensors
        
        print(f"GPU {rank}: Full result matrix C = \n{C_full}")  # GPU tensor displayed
        
        # GPU: On rank 0, verify the result using direct GPU matmul
        if rank == 0:
            expected = torch.matmul(A_full.cuda(), B_full.cuda())  # CPU → GPU, then compute full matmul
            print(f"\nExpected result (computed directly):\n{expected}")
            print(f"Distributed computation matches: {torch.allclose(C_full, expected)}")
    
    finally:
        # Cleanup: Destroy the distributed process group (on CPU)
        if dist.is_initialized():
            dist.destroy_process_group()

def init_process(rank, size, fn, backend='nccl'):
    """Initialize the distributed environment for a single process."""
    # CPU: Set master address and port for all processes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    # GPU: Set this process to use its assigned GPU
    torch.cuda.set_device(rank)
    
    # CPU: Initialize the process group (NCCL backend for GPU communication)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    # CPU: Main process logic and checks
    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)
    
    if torch.cuda.device_count() < 2:
        print(f"Need 2 GPUs, but only {torch.cuda.device_count()} available")
        sys.exit(1)
    
    world_size = 2
    processes = []  # CPU: Store child process handles here
    
    # CPU: Set multiprocessing start method
    if "google.colab" in sys.modules:
        print("Running in Google Colab")
        mp.get_context("spawn")
    else:
        mp.set_start_method("spawn")
    
    # CPU: Spawn two processes; “Spawning” just means: creating a new, independent Python process. Two independent CPU processes, each tied to one GPU rank.
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))  # Assign GPU and task
        p.start()
        processes.append(p)  # Save handle for join()

    # CPU: Wait for all processes to complete
    for p in processes:
        p.join()
