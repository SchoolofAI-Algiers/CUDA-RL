from __future__ import annotations

from datasets import Dataset
from typing import Any
import logging
from finetuning.prompt import CUDA_SYSTEM_PROMPT, build_user_prompt

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
        task_name = example.get('task_name', 'CUDA_Task')
        user_prompt = build_user_prompt(task_name, example['CUDA_Code'])
        prompt = f"{CUDA_SYSTEM_PROMPT}\n\n{user_prompt}"
        
        return {
            "prompt": prompt,
            "prompt": prompt,
            "cuda_runtime": example["CUDA_Runtime"],
            "speedup_native": example["CUDA_Speedup_Native"],
            "speedup_compile": example["CUDA_Speedup_Compile"],
            "original_code": example["CUDA_Code"],
        }
    
    formatted_dataset = dataset.map(_format, remove_columns=dataset.column_names)
    logger.info(f"Formatted {len(formatted_dataset)} samples for GRPO training")
    return formatted_dataset
