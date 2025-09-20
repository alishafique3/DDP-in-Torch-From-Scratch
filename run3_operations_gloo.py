"""run.py:"""
#!/usr/bin/env python
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# def run(rank, size):
#     """ Distributed function to be implemented later. """
#     print("hello")
#     pass

# """Blocking point-to-point communication."""
# def run(rank, size):
#     tensor = torch.zeros(1)
#     if rank == 0:
#         tensor += 1
#         # Send the tensor to process 1
#         dist.send(tensor=tensor, dst=1)
#     else:
#         # Receive tensor from process 0
#         dist.recv(tensor=tensor, src=0)
#     print('Rank ', rank, ' has data ', tensor[0])

# """Non-blocking point-to-point communication."""
# def run(rank, size):
#     tensor = torch.zeros(1)
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
#     """ Reduce example. """
#     group = dist.new_group([0, 1])
#     if rank == 0: # Each rank has different data
#         tensor = torch.tensor([2.0])
#     else:
#         tensor = torch.tensor([3.0])
#     print(f'Rank {rank} before reduce: {tensor[0]}')
    
#     dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=group) # Reduce to rank 0 only, group will allow subset of processes to do the job
    
#     print(f'Rank {rank} after reduce: {tensor[0]}')

# """ All-Reduce example."""
# def run(rank, size):
#     """ Simple collective communication. """
#     group = dist.new_group([0, 1]) # A group is a subset of all our processes. 
#     tensor = torch.ones(1)
#     dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
#     print('Rank ', rank, ' has data ', tensor[0])

# def run(rank, size):
#     """ Gather example. """
#     if rank == 0: # Each rank has different data
#         tensor = torch.tensor([2.0])
#         gather_list = [torch.zeros(1) for _ in range(size)] # Only dst rank needs gather_list
#     else:
#         tensor = torch.tensor([3.0])
#         gather_list = None
    
#     print(f'Rank {rank} before gather: {tensor[0]}')
    
#     dist.gather(tensor, gather_list, dst=0) # Gather all tensors to rank 0
    
#     if rank == 0:
#         print(f'Rank {rank} gathered: {[t[0].item() for t in gather_list]}')
#     else:
#         print(f'Rank {rank} after gather: {tensor[0]}') # Unchanged

# def run(rank, size):
#     """ All-gather example. """
#     if rank == 0: # Each rank has different data
#         tensor = torch.tensor([2.0])
#     else:
#         tensor = torch.tensor([3.0])
    
#     gather_list = [torch.zeros(1) for _ in range(size)] # All ranks need gather_list
    
#     print(f'Rank {rank} before all_gather: {tensor[0]}')
    
#     dist.all_gather(gather_list, tensor) # All ranks get all tensors
    
#     print(f'Rank {rank} gathered: {[t[0].item() for t in gather_list]}')


def run(rank, size):
    """ Broadcast example. """
    if rank == 0: # Only source rank has the data
        tensor = torch.tensor([5.0])
    else:
        tensor = torch.zeros(1) # Other ranks start with zeros
    
    print(f'Rank {rank} before broadcast: {tensor[0]}')
    
    dist.broadcast(tensor, src=0) # Broadcast from rank 0 to all ranks
    
    print(f'Rank {rank} after broadcast: {tensor[0]}')




def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    world_size = 2
    processes = []
    if "google.colab" in sys.modules:
        print("Running in Google Colab")
        mp.get_context("spawn")
    else:
        mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()