# DDP-in-Torch-From-Scratch

A comprehensive implementation of distributed training fundamentals in PyTorch, featuring manual Distributed Data Parallel (DDP) implementation without relying on PyTorch's built-in DDP module. This repository serves as both an educational resource for understanding distributed training internals and a practical implementation for training large language models across multiple GPUs.

# 🎯 Overview

This project demonstrates:

- Low-level distributed communication using NCCL backend
- Manual gradient synchronization across multiple GPUs
- Scalable training from 2 to 8+ GPUs
- Production-ready optimizations for memory efficiency
- LLM training with GPT-2 and Llama 3.2 models

# Hardware Requirements

- GPUs: NVIDIA GPUs with CUDA 11.8+
- Memory: Minimum 16GB per GPU (48GB recommended for Llama)
- Network: InfiniBand or high-speed Ethernet for multi-node setups
- Tested on: A40 (48GB), V100 (32GB), RTX 4090 (24GB)

# Core Implementation
### Manual Gradient Synchronization
Our implementation replaces PyTorch's automatic gradient synchronization with explicit communication:
```python
def all_reduce_gradients(model: nn.Module, world_size: int) -> None:
    """
    Manually synchronize gradients across all processes.
    
    Args:
        model: PyTorch model with computed gradients
        world_size: Number of distributed processes
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                # Sum gradients across all processes
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                # Average by dividing by world size
                param.grad.data.div_(world_size)
```

### Distributed Data Loading
```python
class DistributedSampler:
    """Custom distributed sampler for splitting data across processes."""
    
    def __init__(self, dataset, num_replicas: int, rank: int, shuffle: bool = True):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_samples = len(dataset) // num_replicas
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        
        # Distribute indices across processes
        return iter(indices[self.rank::self.num_replicas])
```

### Training Loop Architecture
```python
def training_step(model, batch, optimizer, world_size):
    """Single training step with manual gradient synchronization."""
    
    # Forward pass (independent on each GPU)
    outputs = model(**batch)
    loss = outputs.loss
    
    # Backward pass (compute local gradients)
    optimizer.zero_grad()
    loss.backward()
    
    # Synchronize gradients across all GPUs
    all_reduce_gradients(model, world_size)
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Update parameters (all GPUs now have identical gradients)
    optimizer.step()
    
    return loss.item()
```
# Conclusion
This project is a minimal but complete example of how **manual DDP training** works in PyTorch.  
Clone it, run it, and use it as a learning resource or as a base for your own experiments.


