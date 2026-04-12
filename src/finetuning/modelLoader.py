from __future__ import annotations
import os
from typing import Any

from unsloth import FastLanguageModel
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
    dtype: Any,
    load_in_4bit: bool,
) -> tuple[Any, Any]:
    os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    logger.info(f"Loaded model and tokenizer for {model_name} | Max seq length: {max_seq_length} | Dtype: {dtype} | Load in 4-bit: {load_in_4bit}")
    return model, tokenizer


def get_peft_model(
    model: Any,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: str,
    lora_target_modules: list[str],
    use_gradient_checkpointing: str,
    use_rslora: bool,
    loftq_config: Any,
    random_state: int,
) -> Any:
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=lora_target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
        use_rslora=use_rslora,
        loftq_config=loftq_config,
    )
    logger.info("PEFT model initialized.")
    model.print_trainable_parameters()
    return model