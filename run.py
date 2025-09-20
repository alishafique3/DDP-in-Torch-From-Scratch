"""run.py:"""
#!/usr/bin/env python
import os
import sys
import torch
import torch.distributed as dist           # CPU: Used for setting up distributed communication
import torch.multiprocessing as mp         # CPU: Used for spawning multiple independent processes

def run(rank, size):
    """ 
    CPU: Function executed by each spawned process.

    - rank: unique ID of this process (0, 1, ..., world_size-1)
    - size: total number of processes (world size)
    """
    print("hello")  # CPU: This will be printed by each process independently
    pass  # Placeholder for real distributed work

def init_process(rank, size, fn, backend='gloo'):
    """ 
    CPU: Initializes the distributed process group.

    - Sets environment variables for inter-process communication.
    - Uses the Gloo backend (CPU-based).
    - Calls the main distributed function (`fn`) with this process's rank.
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'    # CPU: Localhost for single-machine setup
    os.environ['MASTER_PORT'] = '29500'        # CPU: Port used for all processes to connect

    dist.init_process_group(backend, rank=rank, world_size=size)  # CPU: Initialize distributed context
    fn(rank, size)  # CPU: Run the actual distributed function (on CPU)

if __name__ == "__main__":
    world_size = 2             # CPU: Number of total processes (can represent 2 workers)
    processes = []             # CPU: List to keep track of spawned processes

    # CPU: Set multiprocessing start method depending on environment
    if "google.colab" in sys.modules:
        print("Running in Google Colab")
        mp.get_context("spawn")                # CPU: Get context safely in Colab (spawn is default)
    else:
        mp.set_start_method("spawn")           # CPU: Required for PyTorch multiprocessing

    # CPU: Spawn one process per rank (rank 0 and rank 1)
    for rank in range(world_size):
        # CPU: Create new process and assign it a rank and role
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()                              # CPU: Start process execution
        processes.append(p)                    # CPU: Track process for synchronization

    # CPU: Wait for all spawned processes to finish
    for p in processes:
        p.join()                               # CPU: Block until the process completes
