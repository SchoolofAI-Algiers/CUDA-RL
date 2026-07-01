from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable
import logging

from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_mock_modules() -> None:
    """
    Mock required modules that may not be installed.
    This allows the code to run in environments where these modules are optional.
    """
    from unittest.mock import MagicMock
    
    sys.modules['mergekit'] = MagicMock()
    sys.modules['mergekit.config'] = MagicMock()
    sys.modules['mergekit.merge'] = MagicMock()
    sys.modules['llm_blender'] = MagicMock()
    sys.modules['weave'] = MagicMock()
    sys.modules['weave.trace'] = MagicMock()
    sys.modules['weave.trace.context'] = MagicMock()
    
    logger.info("Mocked optional modules for compatibility")


def build_grpo_trainer(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    reward_func: Callable,
    grpo_config: GRPOConfig,
) -> GRPOTrainer:
    """
    Build a GRPO trainer instance.
    
    Args:
        model: The language model to train.
        tokenizer: The tokenizer corresponding to the model.
        dataset: Training dataset.
        reward_func: Reward function for the model outputs.
        grpo_config: GRPO configuration object.
        
    Returns:
        Initialized GRPOTrainer instance.
    """
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=dataset,
    )
    
    logger.info("GRPO trainer built successfully")
    return trainer


def run_grpo_training(trainer: GRPOTrainer) -> Any:
    """
    Run GRPO training.
    
    Args:
        trainer: The GRPOTrainer instance.
        
    Returns:
        Training results.
    """
    logger.info("Starting GRPO training...")
    results = trainer.train()
    logger.info("GRPO training completed")
    return results


def save_model_artifacts(
    model: Any,
    tokenizer: Any,
    output_dir: str,
) -> None:
    """
    Save model and tokenizer artifacts.
    
    Args:
        model: Trained model.
        tokenizer: Tokenizer.
        output_dir: Directory to save artifacts.
    """
    output_path = Path(output_dir) / "final_model"
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    logger.info(f"Model and tokenizer saved to {output_path}")
