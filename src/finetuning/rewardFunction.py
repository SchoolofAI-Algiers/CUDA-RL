from __future__ import annotations
from typing import Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def cuda_optimization_reward(
    completions: list[str],
    cuda_runtime: list[float] | None = None,
    speedup_native: list[float] | None = None,
    speedup_compile: list[float] | None = None,
    **kwargs: Any,
) -> list[float]:
    rewards = []
    n = len(completions)
    
    # Use default values if not provided
    cuda_rt = cuda_runtime if cuda_runtime else [0.0] * n
    sp_nat = speedup_native if speedup_native else [1.0] * n
    sp_com = speedup_compile if speedup_compile else [1.0] * n
    
    for i, comp in enumerate(completions):
        base_reward = 0.6 * sp_nat[i] + 0.4 * sp_com[i]
        rewards.append(base_reward)
    
    logger.debug(f"Computed rewards for {len(completions)} completions. Mean reward: {sum(rewards) / len(rewards):.4f}")
    return rewards
