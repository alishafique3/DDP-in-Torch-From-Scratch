# DDP-in-Torch-From-Scratch

A comprehensive implementation of distributed training fundamentals in PyTorch, featuring manual Distributed Data Parallel (DDP) implementation without relying on PyTorch's built-in DDP module. This repository serves as both an educational resource for understanding distributed training internals and a practical implementation for training large language models across multiple GPUs.

# Overview

This project demonstrates:

- Low-level distributed communication using NCCL backend
- Manual gradient synchronization across multiple GPUs
- Scalable training from 2 to 8+ GPUs
- Production-ready optimizations for memory efficiency
- LLM training with GPT-2 and Llama 3.2 models
