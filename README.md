# Distributed Training Tutorial

This repository contains theory and hands-on materials for the tutorial on Distributed Training, covering key concepts and practical exercises using PyTorch Distributed Data Parallel (DDP) and DeepSpeed for scalable AI/LLM training.

## Topics Covered
1. **Distributed Training Concepts**:
   - Vertical vs. Horizontal Scaling
   - Data, Model, and Pipeline Parallelism
2. **Hands-On**:
   - Multi-GPU Training with PyTorch DDP
   - Scalable AI/LLM Training with DeepSpeed
   - Using SLURM for Job Scheduling

## Prerequisites
- Python 3.9+
- NVIDIA GPUs and CUDA toolkit
- PyTorch, DeepSpeed, and SLURM installed

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/sowmyashree04/distributed_training.git
   cd distributed-training-workshop

   
## Code Progression for Hands On session

Each code file builds upon the previous one, starting with a non-distributed script for single GPU training and progressively advancing to multi-node training on a Slurm cluster.

### Files Overview

- **`single_gpu.py`**  
  Non-distributed training script for single GPU.

- **`multigpu.py`**  
  Implements Distributed Data Parallel (DDP) on a single node.

- **`multigpu_torchrun.py`**  
  Demonstrates single-node DDP training using Torchrun.

- **`multinode.py`**  
  Extends DDP to multiple nodes using Torchrun, with optional integration into Slurm.

### Slurm Setup

- **`slurm/setup_pcluster_slurm.md`**  
  Instructions for setting up an AWS cluster.

- **`slurm/config.yaml.template`**  
  Template configuration file for AWS cluster setup.

- **`slurm/sbatch_run.sh`**  
  Slurm script to submit and launch the distributed training job.
