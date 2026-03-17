# RL for CUDA Kernel Optimization

Reinforcement Learning framework for automatic optimization of CUDA kernels. The project models kernel transformation as an MDP where an agent learns to apply optimizations (fusion, fission, unrolling) to maximize execution speed.

## Features

* RL-based kernel optimization 
* Custom CUDA environment with performance feedback
* Focus on key benchmarks (MatMul, element-wise ops, reductions)


## Goal

Discover optimization strategies that outperform heuristic or manual tuning.