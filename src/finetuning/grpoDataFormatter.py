from __future__ import annotations

from datasets import Dataset
from typing import Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_cuda_dataset_for_grpo(dataset: Dataset) -> Dataset:
    """
    Format CUDA dataset specifically for GRPO training.
    
    Args:
        dataset: Raw CUDA dataset from HuggingFace.
        
    Returns:
        Formatted dataset with GRPO-specific fields.
    """
    def _format(example: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            f"Optimize the following CUDA kernel for maximum performance while maintaining correctness.\n\n"
            f"```cuda\n{example['CUDA_Code']}\n```\n\n"
            f"Return ONLY the optimized CUDA code. Do not include explanations, markdown, or extra text."
        )
        return {
            "prompt": prompt,
            "cuda_runtime": example["CUDA_Runtime"],
            "speedup_native": example["CUDA_Speedup_Native"],
            "speedup_compile": example["CUDA_Speedup_Compile"],
            "original_code": example["CUDA_Code"],
        }
    
    formatted_dataset = dataset.map(_format, remove_columns=dataset.column_names)
    logger.info(f"Formatted {len(formatted_dataset)} samples for GRPO training")
    return formatted_dataset
