from __future__ import annotations

import warnings

import torch


def suppress_unsloth_warnings() -> None:
    warnings.filterwarnings("ignore", message="Unable to fetch remote file")
    warnings.filterwarnings("ignore", message="Could not find a config file")


def print_gpu_stats() -> dict[str, float]:
    if not torch.cuda.is_available():
        print("No CUDA device found — skipping GPU stats.")
        return {}

    props = torch.cuda.get_device_properties(0)
    total_gb = round(props.total_memory / 1024 ** 3, 3)
    reserved_gb = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)

    print(
        f"GPU: {props.name}  |  "
        f"Total VRAM: {total_gb} GB  |  "
        f"Reserved: {reserved_gb} GB"
    )
    return {"name": props.name, "total_gb": total_gb, "reserved_gb": reserved_gb}


def print_peak_memory(start_reserved_gb: float) -> None:
    if not torch.cuda.is_available():
        return

    peak_gb = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    used_gb = round(peak_gb - start_reserved_gb, 3)
    print(f"Peak VRAM: {peak_gb} GB  |  Used for training: {used_gb} GB")