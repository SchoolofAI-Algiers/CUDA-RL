from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import yaml
import wandb
from trl import GRPOConfig

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetuning.datasetLoader import load_cuda_dataset
from finetuning.modelLoader import load_model_and_tokenizer, get_peft_model
from finetuning.grpoTrainer import (
    setup_mock_modules,
    build_grpo_trainer,
    run_grpo_training,
    save_model_artifacts,
)
from finetuning.grpoDataFormatter import format_cuda_dataset_for_grpo
from finetuning.rewardFunction import cuda_optimization_reward
from finetuning.gpuStats import print_gpu_stats, suppress_unsloth_warnings


DEFAULT_CONFIG = "./config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training script for CUDA kernel optimization")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="cuda-kernel-optimization",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="cuda_opt_grpo_v1",
        help="Weights & Biases run name",
    )

    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_grpo_config_from_dict(cfg: dict) -> GRPOConfig:
    """Build GRPOConfig directly from configuration dictionary."""
    return GRPOConfig(
        output_dir=cfg.get("grpo_output_dir", "./qwen25_cuda_grpo"),
        run_name=cfg.get("grpo_run_name", "cuda_opt_grpo_v1"),
        learning_rate=cfg.get("grpo_learning_rate", 2e-5),
        num_train_epochs=cfg.get("grpo_num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("grpo_per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg.get("grpo_gradient_accumulation_steps", 4),
        max_prompt_length=cfg.get("grpo_max_prompt_length", 1024),
        max_completion_length=cfg.get("grpo_max_completion_length", 1024),
        num_generations=cfg.get("grpo_num_generations", 4),
        temperature=cfg.get("grpo_temperature", 0.7),
        top_p=cfg.get("grpo_top_p", 0.9),
        logging_steps=cfg.get("grpo_logging_steps", 1),
        save_strategy=cfg.get("grpo_save_strategy", "steps"),
        save_steps=cfg.get("grpo_save_steps", 200),
        save_total_limit=cfg.get("grpo_save_total_limit", 3),
        bf16=cfg.get("grpo_bf16", True),
        report_to=cfg.get("grpo_report_to", "wandb"),
        beta=cfg.get("grpo_beta", 0.04),
        epsilon=cfg.get("grpo_epsilon", 0.2),
    )


def main() -> None:
    # Suppress warnings
    warnings.filterwarnings("ignore")
    suppress_unsloth_warnings()
    
    # Parse arguments and load config
    args = parse_args()
    cfg = load_config(args.config)
    
    # Setup mock modules for optional dependencies
    setup_mock_modules()
    
    # Print GPU stats
    print_gpu_stats()
    
    # Initialize Weights & Biases if requested
    if args.use_wandb:
        wandb.login()
        wandb.init(
            project=args.project_name,
            name=args.run_name,
            notes="GRPO training for CUDA kernel optimization"
        )
    
    # Load dataset
    dataset = load_cuda_dataset(
        dataset_name=cfg["dataset_name"],
        splits=cfg["splits"],
        correct_only=cfg["correct_only"],
        num_samples=cfg["num_samples"],
    )
    
    # Format dataset for GRPO
    dataset = format_cuda_dataset_for_grpo(dataset)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=cfg["dtype"],
        load_in_4bit=cfg["load_in_4bit"],
    )
    
    # Apply LoRA
    model = get_peft_model(
        model=model,
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        lora_bias=cfg["lora_bias"],
        lora_target_modules=cfg["lora_target_modules"],
        use_gradient_checkpointing=cfg["use_gradient_checkpointing"],
        use_rslora=cfg["use_rslora"],
        loftq_config=cfg["loftq_config"],
        random_state=cfg["random_state"],
    )
    
    # Build GRPO config
    grpo_config = build_grpo_config_from_dict(cfg)
    
    # Build trainer
    trainer = build_grpo_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_func=cuda_optimization_reward,
        grpo_config=grpo_config,
    )
    
    # Run training
    results = run_grpo_training(trainer)
    
    # Save model artifacts
    save_model_artifacts(
        model=model,
        tokenizer=tokenizer,
        output_dir=grpo_config.output_dir,
    )
    
    # Log to wandb if enabled
    if args.use_wandb:
        wandb.log({
            "final_model_path": grpo_config.output_dir,
            "training_samples": len(dataset),
            "max_sequence_length": cfg["max_seq_length"],
        })
        wandb.finish()
    
    print("GRPO training completed successfully!")


if __name__ == "__main__":
    main()
