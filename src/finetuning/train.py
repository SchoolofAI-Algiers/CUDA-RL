

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasetLoader import format_dataset, load_cuda_dataset
from modelLoader import get_peft_model, load_model_and_tokenizer
from trainer import build_trainer, run_training
from gpuStats import print_gpu_stats, print_peak_memory, suppress_unsloth_warnings


DEFAULT_CONFIG = "./config.yaml"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning script")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the YAML config file (default: config.yaml)",
    )

    return parser.parse_args()

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    suppress_unsloth_warnings()

    model, tokenizer = load_model_and_tokenizer(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=cfg["dtype"],
        load_in_4bit=cfg["load_in_4bit"],
    )

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


    dataset = load_cuda_dataset(
        dataset_name=cfg["dataset_name"],
        splits=cfg["splits"],
        correct_only=cfg["correct_only"],
        num_samples=cfg.get("num_samples"),
    )
    dataset = format_dataset(dataset, eos_token=tokenizer.eos_token , columns=cfg["format_columns"])


    stats = print_gpu_stats()
    start_reserved = stats.get("reserved_gb", 0.0)


    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_seq_length=cfg["max_seq_length"],
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_steps=cfg["warmup_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        optim=cfg["optim"],
        logging_steps=cfg["logging_steps"],
        seed=cfg["seed"],
        packing=cfg["packing"],
        dataset_num_proc=cfg["dataset_num_proc"],
    )

    run_training(trainer)
    print_peak_memory(start_reserved)


if __name__ == "__main__":
    main()