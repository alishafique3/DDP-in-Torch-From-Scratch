#!/usr/bin/env python
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# def run(rank, size):
#     """ Hello world example. """
#     torch.cuda.set_device(rank)
#     device = torch.cuda.current_device()
#     print(f"Hello from rank {rank} on GPU {device}")



# def run(rank, size):
#     """ Blocking point-to-point communication with NCCL. """
#     torch.cuda.set_device(rank)
#     device = torch.cuda.current_device()
#     
#     tensor = torch.zeros(1).to(device)
#     if rank == 0:
#         tensor += 1
#         # Send the tensor to process 1
#         dist.send(tensor=tensor, dst=1)
#     else:
#         # Receive tensor from process 0
#         dist.recv(tensor=tensor, src=0)
#     print('Rank ', rank, ' has data ', tensor[0])




# def run(rank, size):
#     """ Non-blocking point-to-point communication with NCCL. """
#     torch.cuda.set_device(rank)
#     device = torch.cuda.current_device()
#     
#     tensor = torch.zeros(1).to(device)
#     req = None
#     if rank == 0:
#         tensor += 1
#         # Send the tensor to process 1
#         req = dist.isend(tensor=tensor, dst=1)
#         print('Rank 0 started sending')
#     else:
#         # Receive tensor from process 0
#         req = dist.irecv(tensor=tensor, src=0)
#         print('Rank 1 started receiving')
#     req.wait() # block and wait till non blocking communication completed
#     print('Rank ', rank, ' has data ', tensor[0])




# def run(rank, size):
#     """ Reduce example with NCCL. """
#     torch.cuda.set_device(rank)
#     device = torch.cuda.current_device()
#     
#     if rank == 0: # Each rank has different data
#         tensor = torch.tensor([2.0]).to(device)
#     else:
#         tensor = torch.tensor([3.0]).to(device)
#     print(f'Rank {rank} before reduce: {tensor[0]}')
#     
#     dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM) # Reduce to rank 0 only
#     
#     print(f'Rank {rank} after reduce: {tensor[0]}')




# def run(rank, size):
#     """ All-Reduce example with NCCL. """
#     torch.cuda.set_device(rank)
#     device = torch.cuda.current_device()
#     
#     tensor = torch.ones(1).to(device)
#     print(f'Rank {rank} before all-reduce: {tensor[0]}')
#     
#     dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
#     
#     print('Rank ', rank, ' has data ', tensor[0])




# def run(rank, size):
#     """ Gather example with NCCL. """
#     torch.cuda.set_device(rank)
#     device = torch.cuda.current_device()
#     
#     if rank == 0: # Each rank has different data
#         tensor = torch.tensor([2.0]).to(device)
#         gather_list = [torch.zeros(1).to(device) for _ in range(size)] # Only dst rank needs gather_list
#     else:
#         tensor = torch.tensor([3.0]).to(device)
#         gather_list = None
#     
#     print(f'Rank {rank} before gather: {tensor[0]}')
#     
#     dist.gather(tensor, gather_list, dst=0) # Gather all tensors to rank 0
#     
#     if rank == 0:
#         print(f'Rank {rank} gathered: {[t[0].item() for t in gather_list]}')
#     else:
#         print(f'Rank {rank} after gather: {tensor[0]}') # Unchanged




# def run(rank, size):
#     """ All-gather example with NCCL. """
#     torch.cuda.set_device(rank)
#     device = torch.cuda.current_device()
#     
#     if rank == 0: # Each rank has different data
#         tensor = torch.tensor([2.0]).to(device)
#     else:
#         tensor = torch.tensor([3.0]).to(device)
#     
#     gather_list = [torch.zeros(1).to(device) for _ in range(size)] # All ranks need gather_list
#     
#     print(f'Rank {rank} before all_gather: {tensor[0]}')
#     
#     dist.all_gather(gather_list, tensor) # All ranks get all tensors
#     
#     print(f'Rank {rank} gathered: {[t[0].item() for t in gather_list]}')




def run(rank, size):
    """ Broadcast example with NCCL. """
    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()
    
    if rank == 0: # Only source rank has the data
        tensor = torch.tensor([5.0]).to(device)
    else:
        tensor = torch.zeros(1).to(device) # Other ranks start with zeros
    
    print(f'Rank {rank} before broadcast: {tensor[0]}')
    
    dist.broadcast(tensor, src=0) # Broadcast from rank 0 to all ranks
    
    print(f'Rank {rank} after broadcast: {tensor[0]}')




def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

    try:
        fn(rank, size)
    finally:
        dist.destroy_process_group()  # Clean up process group




if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)
    
    world_size = 2
    processes = []
    mp.set_start_method("spawn")
    
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()