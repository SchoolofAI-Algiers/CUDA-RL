from __future__ import annotations
from datasets import Dataset, concatenate_datasets, load_dataset
import yaml
from typing import Any

# template 
# Todo: Shall be refined to include more specific instructions
CUDA_RL_TASK = """\
Below is an instruction that describes a task. Write a response that appropriately completes the request.
 
### Instruction:
You are an expert CUDA kernel engineer. Given a PyTorch module implementation, write an optimized CUDA kernel that is functionally equivalent but achieves maximum GPU performance.
 
Operation: {Op_Name}
 
### Input:
{PyTorch_Code_Module}
 
### Response:
{CUDA_Code}"""


# load the data
def load_cuda_dataset(
    dataset_name: str,
    splits : list[str],
    correct_only: bool = True,
    num_samples: int | None = None,
) -> Dataset:

    raw = load_dataset(dataset_name)
 
    filtered_splits: list[Dataset] = []
    for split in splits :
        split = raw[split]
        if correct_only:
            split = split.filter(lambda x: x["Correct"] is True)
        filtered_splits.append(split)
 
    dataset = concatenate_datasets(filtered_splits)
 
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    return dataset
 

def format_dataset(dataset: Dataset, eos_token: str , columns: list[str]) -> Dataset:
    def _format(examples: dict[str, Any]) -> dict[str, list[str]]:
        texts = []
        batch_size = len(next(iter(examples.values())))
        for i in range(batch_size):
            row = {}

            for col in columns:
                if col not in examples:
                    raise ValueError(f"Column '{col}' not found in dataset.")
                row[col] = examples[col][i]

            text = CUDA_RL_TASK.format(**row) + eos_token
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(_format, batched=True)
    return dataset