from __future__ import annotations

from typing import Any

from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
import logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_trainer(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    max_seq_length: int,
    output_dir: str,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    warmup_steps: int,
    learning_rate: float,
    weight_decay: float,
    lr_scheduler_type: str,
    optim: str,
    logging_steps: int,
    seed: int,
    packing: bool,
    dataset_num_proc: int,
) -> SFTTrainer:
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=logging_steps,
        optim=optim,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
        output_dir=output_dir,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=dataset_num_proc,
        packing=packing,
        args=training_args,
    )
    return trainer


def run_training(trainer: SFTTrainer) -> Any:
    stats = trainer.train()
    logger.info("Training completed.")
    return stats