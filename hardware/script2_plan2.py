#!/usr/bin/env python3
"""
Script 2 v4: Hardware Evaluation Environment for CUDA Kernel RL Optimization
           — Shape-Aware PyTorch Extension Edition

CHANGES FROM v3:
  - The LLM now emits a @SHAPES: annotation in its comment block.
  - parse_shapes_from_source() extracts it.
  - build_tensors_from_shapes() converts it to real CUDA tensors.
  - _benchmark_subprocess() uses declared shapes first; falls back to
    the old candidate list only when the annotation is absent or malformed.
  - atol raised from 1e-4 → 1e-3 (correct for large FP32 matmuls).
  - candidates_2 expanded with matrix-vector, 4D batched, (N,N)@(N,1).
"""

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from io import StringIO
from pathlib import Path

import pandas as pd
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
PENDING_DIR   = "pending_kernels"
COMPLETED_DIR = "completed_results"

POLL_INTERVAL_SECONDS    = 1.0
RUN_TIMEOUT_SECONDS      = 30.0
PROFILE_TIMEOUT_SECONDS  = 60.0
FILE_STABLE_WAIT_SECONDS = 0.3
KEEP_FAILED_ARTIFACTS    = True

N_WARMUP = 5
N_BENCH  = 50
TENSOR_N = 1 << 20      # fallback for 1D elementwise when no annotation
MATH_TOL = 1e-3         # raised: FP32 large-K matmuls can differ by ~1e-3

CUDA_ARCH = "sm_89"

TIME_METRIC = "gpu__time_duration.sum"
OPTIMIZED_METRICS = [
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
]

# ──────────────────────────────────────────────────────────────────────────────
# Pybind11 / bridge templates  (unchanged from v3)
# ──────────────────────────────────────────────────────────────────────────────
PYBIND_WRAPPER_TEMPLATE = """
#ifndef _PYBIND_WRAPPER_INJECTED
#define _PYBIND_WRAPPER_INJECTED
#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("baseline_forward",  &baseline_forward,  "Baseline kernel (reference)");
    m.def("optimized_forward", &optimized_forward, "Optimized kernel (LLM output)");
}}
#endif
"""

BARE_KERNEL_BRIDGE = """
#include <torch/extension.h>
__global__ void baseline_kernel(float* in, float* out, int N);
__global__ void optimized_kernel(float* in, float* out, int N);

torch::Tensor baseline_forward(torch::Tensor input) {
    auto out = torch::empty_like(input);
    int N    = input.numel();
    int grid = (N + 255) / 256;
    baseline_kernel<<<grid, 256>>>(input.data_ptr<float>(), out.data_ptr<float>(), N);
    return out;
}

torch::Tensor optimized_forward(torch::Tensor input) {
    auto out = torch::empty_like(input);
    int N    = input.numel();
    int grid = (N + 255) / 256;
    optimized_kernel<<<grid, 256>>>(input.data_ptr<float>(), out.data_ptr<float>(), N);
    return out;
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Utilities  (unchanged from v3)
# ──────────────────────────────────────────────────────────────────────────────
def ensure_directories() -> None:
    os.makedirs(PENDING_DIR,   exist_ok=True)
    os.makedirs(COMPLETED_DIR, exist_ok=True)


def log_msg(level: str, message: str) -> None:
    ts     = time.strftime("%Y-%m-%d %H:%M:%S")
    target = sys.stderr if level == "CRITICAL" else sys.stdout
    print(f"[{ts}] [{level}] {message}", file=target, flush=True)


def compact_text(text: str, max_chars: int = 2000) -> str:
    if not text:
        return ""
    stripped = text.strip()
    return stripped if len(stripped) <= max_chars else stripped[:max_chars] + "\n...[truncated]"


def is_file_stable(path: str, wait_seconds: float = FILE_STABLE_WAIT_SECONDS) -> bool:
    if not os.path.exists(path):
        return False
    try:
        before = os.stat(path)
        time.sleep(wait_seconds)
        after  = os.stat(path)
    except FileNotFoundError:
        return False
    return before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns


def acquire_trial_lock(trial_id: str) -> "str | None":
    lock_path = os.path.join(PENDING_DIR, f".{trial_id}.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return lock_path
    except FileExistsError:
        return None


def release_trial_lock(lock_path: "str | None") -> None:
    if not lock_path:
        return
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass


def cleanup_artifacts(paths: list) -> None:
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


def list_trial_ids() -> list:
    trial_ids = []
    for name in os.listdir(PENDING_DIR):
        if name.endswith("_kernel.cu"):
            trial_ids.append(name[: -len("_kernel.cu")])
    trial_ids.sort()
    return trial_ids


def build_result_payload(
    trial_id, status, reward, error_trace,
    baseline_time, optimized_time, occ, comp, mem
) -> dict:
    return {
        "trial_id":    trial_id,
        "status":      status,
        "reward":      float(reward),
        "error_trace": error_trace,
        "metrics": {
            "baseline_time_ns":  float(baseline_time)  if baseline_time  is not None else None,
            "optimized_time_ns": float(optimized_time) if optimized_time is not None else None,
            "occupancy_pct":     float(occ)            if occ            is not None else None,
            "compute_pct":       float(comp)            if comp           is not None else None,
            "memory_pct":        float(mem)            if mem            is not None else None,
        },
    }


def atomic_write_json(result_path: str, payload: dict, trial_id: str) -> None:
    temp_path = os.path.join(COMPLETED_DIR, f"temp_{trial_id}.json")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.rename(temp_path, result_path)


def run_command(
    cmd: list,
    timeout: float = None,
    enforce_process_group: bool = False
) -> subprocess.CompletedProcess:
    if enforce_process_group:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, preexec_fn=os.setsid
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return subprocess.CompletedProcess(
                args=cmd, returncode=process.returncode,
                stdout=stdout, stderr=stderr
            )
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(
                cmd=cmd, timeout=timeout, output=stdout, stderr=stderr
            )
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


# ──────────────────────────────────────────────────────────────────────────────
# Ncu CSV parsing  (unchanged from v3)
# ──────────────────────────────────────────────────────────────────────────────
def parse_ncu_csv_to_dataframe(ncu_stdout: str) -> pd.DataFrame:
    lines = ncu_stdout.splitlines()
    start_index = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('"ID"') or stripped.startswith("ID"):
            start_index = i
            break
    if start_index is None:
        raise ValueError("Failed to locate CSV header in ncu output.")
    csv_text = "\n".join(lines[start_index:])
    return pd.read_csv(StringIO(csv_text), engine="python", on_bad_lines="skip")


def detect_columns(df: pd.DataFrame) -> tuple:
    name_candidates   = ["Metric Name", "Name",   "metric_name"]
    value_candidates  = ["Metric Value", "Value",  "metric_value"]
    kernel_candidates = ["Kernel Name",  "Kernel", "kernel_name"]

    metric_name_col  = next((c for c in name_candidates   if c in df.columns), None)
    metric_value_col = next((c for c in value_candidates  if c in df.columns), None)
    kernel_name_col  = next((c for c in kernel_candidates if c in df.columns), None)

    if any(c is None for c in [metric_name_col, metric_value_col, kernel_name_col]):
        raise ValueError(
            f"Could not detect required ncu CSV columns. Found: {list(df.columns)}"
        )
    return metric_name_col, metric_value_col, kernel_name_col


def extract_metric_for_kernel_name(
    df: pd.DataFrame, kernel_name: str, metric_name: str
) -> float:
    metric_name_col, metric_value_col, kernel_name_col = detect_columns(df)
    matched = df[
        (df[kernel_name_col].astype(str) == str(kernel_name)) &
        (df[metric_name_col] == metric_name)
    ]
    if matched.empty:
        raise ValueError(f"Metric '{metric_name}' for kernel '{kernel_name}' not found.")
    raw_value = matched.iloc[0][metric_value_col]
    return float(str(raw_value).strip().replace(",", ""))


def choose_kernel_names(df: pd.DataFrame) -> tuple:
    metric_name_col, _, kernel_name_col = detect_columns(df)
    time_rows = df[df[metric_name_col] == TIME_METRIC]
    if time_rows.empty:
        raise ValueError(f"No '{TIME_METRIC}' rows found in ncu output.")

    kernel_series   = time_rows[kernel_name_col].astype(str)
    baseline_cands  = kernel_series[kernel_series.str.contains("baseline",  case=False, na=False)]
    optimized_cands = kernel_series[kernel_series.str.contains("optimized", case=False, na=False)]

    if not baseline_cands.empty and not optimized_cands.empty:
        return baseline_cands.iloc[0], optimized_cands.iloc[0]

    ordered = list(dict.fromkeys(kernel_series.tolist()))
    if len(ordered) >= 2:
        return ordered[0], ordered[-1]

    raise ValueError("Unable to disambiguate baseline/optimized kernels from ncu output.")


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Shape annotation parsing
# ──────────────────────────────────────────────────────────────────────────────

# Supported dtypes mapping: annotation string → torch dtype
_DTYPE_MAP = {
    "float32": "torch.float32",
    "float16": "torch.float16",
    "int32":   "torch.int32",
    "int64":   "torch.int64",
}

# Regex: matches e.g.  float32(1024, 512)  or  float32(1048576)
_TENSOR_SPEC_RE = re.compile(
    r"(float32|float16|int32|int64)\(([^)]+)\)"
)


# def parse_shapes_from_source(cuda_src: str) -> "list[dict] | None":
#     """
#     Scan the CUDA source for a line of the form:
#         * @SHAPES: float32(1024, 1024) | float32(1024, 512)
#     Returns a list of dicts like:
#         [{"dtype": "torch.float32", "shape": [1024, 1024]},
#          {"dtype": "torch.float32", "shape": [1024, 512]}]
#     Returns None if no valid annotation is found.
#     """
#     for line in cuda_src.splitlines():
#         if "@SHAPES:" not in line:
#             continue
#         # Extract the part after @SHAPES:
#         spec_str = line.split("@SHAPES:", 1)[1].strip().rstrip("*/").strip()
#         parts    = [p.strip() for p in spec_str.split("|")]
#         tensors  = []
#         ok       = True
#         for part in parts:
#             m = _TENSOR_SPEC_RE.match(part.strip())
#             if not m:
#                 log_msg("WARN", f"Could not parse shape spec part: '{part}' — falling back to candidates.")
#                 ok = False
#                 break
#             dtype_str = m.group(1)
#             dims      = [int(d.strip()) for d in m.group(2).split(",")]
#             tensors.append({"dtype": _DTYPE_MAP[dtype_str], "shape": dims})
#         if ok and tensors:
#             return tensors
#     return None

MAX_TENSOR_ELEMENTS = 64 * 1024 * 1024  # 64M floats = 256 MB

def parse_shapes_from_source(cuda_src: str) -> "list[dict] | None":
    """
    Scan the CUDA source for a @SHAPES: annotation and parse it.
    Returns None on ANY parse failure — never raises.
    Rejects any spec where a single tensor exceeds MAX_TENSOR_ELEMENTS.
    """
    for line in cuda_src.splitlines():
        if "@SHAPES:" not in line:
            continue
        spec_str = line.split("@SHAPES:", 1)[1].strip().rstrip("*/").strip()
        parts    = [p.strip() for p in spec_str.split("|")]
        tensors  = []
        ok       = True
        for part in parts:
            m = _TENSOR_SPEC_RE.match(part.strip())
            if not m:
                log_msg("WARN", f"@SHAPES part '{part}' did not match expected pattern — falling back.")
                ok = False
                break
            dtype_str = m.group(1)
            # ── Safe integer parsing: reject symbolic names like M, K, N ─────
            raw_dims = [d.strip() for d in m.group(2).split(",")]
            dims = []
            for raw in raw_dims:
                try:
                    dims.append(int(raw))
                except ValueError:
                    log_msg("WARN",
                        f"@SHAPES contains non-integer dimension '{raw}' "
                        f"(LLM used a symbolic name instead of a concrete integer) — falling back."
                    )
                    ok = False
                    break
            if not ok:
                break
            # ── Reject oversized tensors ──────────────────────────────────────
            n_elements = 1
            for d in dims:
                n_elements *= d
            if n_elements > MAX_TENSOR_ELEMENTS:
                log_msg("WARN",
                    f"Declared shape {dtype_str}{tuple(dims)} has {n_elements:,} elements "
                    f"(>{MAX_TENSOR_ELEMENTS:,} limit) — rejecting, using fallback."
                )
                return None
            tensors.append({"dtype": _DTYPE_MAP[dtype_str], "shape": dims})
        if ok and tensors:
            return tensors
    return None

def build_tensors_from_spec(
    tensor_specs: "list[dict]",
    device: str = "cuda"
) -> "list":
    """
    Convert parsed shape specs into actual torch tensors on the given device.
    Returns a list of tensors (1 or 2 elements) ready to pass as *args.
    """
    import torch
    tensors = []
    for spec in tensor_specs:
        dtype = eval(spec["dtype"])   # e.g. torch.float32
        shape = spec["shape"]
        if dtype in (torch.int32, torch.int64):
            t = torch.randint(0, 10, shape, dtype=dtype, device=device)
        else:
            t = torch.rand(shape, dtype=dtype, device=device)
        tensors.append(t)
    return tensors


def shape_spec_to_python_literal(tensor_specs: "list[dict]") -> str:
    """
    Serialise parsed shape specs to a Python literal string that can be
    embedded verbatim into the benchmark subprocess driver.

    Example output:
        [{"dtype": "torch.float32", "shape": [1024, 1024]},
         {"dtype": "torch.float32", "shape": [1024, 512]}]
    """
    return json.dumps(tensor_specs)


# ──────────────────────────────────────────────────────────────────────────────
# Source preparation  (same logic as v3 + alias injection)
# ──────────────────────────────────────────────────────────────────────────────
def _detect_code_style(cuda_src: str) -> str:
    has_torch_tensor = "torch::Tensor" in cuda_src
    has_at_dispatch  = "AT_DISPATCH"   in cuda_src
    has_bare_kernels = (
        "__global__" in cuda_src and
        "baseline_kernel"  in cuda_src and
        "optimized_kernel" in cuda_src
    )
    if has_torch_tensor or has_at_dispatch:
        return "torch_ext"
    if has_bare_kernels:
        return "bare_cuda"
    return "torch_ext"


def _find_forward_functions(cuda_src: str) -> list:
    """Return all torch::Tensor function names defined in the source."""
    pattern = r'torch::Tensor\s+(\w+)\s*\('
    return re.findall(pattern, cuda_src)


def _prepare_cuda_source(cuda_src: str) -> str:
    style = _detect_code_style(cuda_src)

    if "#include <torch/extension.h>" not in cuda_src:
        cuda_src = "#include <torch/extension.h>\n" + cuda_src

    if style == "bare_cuda":
        if "PYBIND11_MODULE" not in cuda_src:
            cuda_src = cuda_src + "\n" + BARE_KERNEL_BRIDGE
            cuda_src = cuda_src + "\n" + PYBIND_WRAPPER_TEMPLATE.format()
        return cuda_src

    # torch_ext style — ensure baseline_forward / optimized_forward exist
    defined_fns   = _find_forward_functions(cuda_src)
    has_baseline  = "baseline_forward"  in defined_fns
    has_optimized = "optimized_forward" in defined_fns

    alias_block = ""
    if not has_baseline or not has_optimized:
        candidates = [f for f in defined_fns if "forward" in f.lower()]
        if not candidates:
            candidates = defined_fns
        if candidates:
            fn_name = candidates[0]
            if not has_baseline:
                alias_block += (
                    f"\n// AUTO-ALIAS: baseline_forward → {fn_name}\n"
                    f"torch::Tensor baseline_forward(torch::Tensor x) "
                    f"{{ return {fn_name}(x); }}\n"
                )
            if not has_optimized:
                alias_block += (
                    f"\n// AUTO-ALIAS: optimized_forward → {fn_name}\n"
                    f"torch::Tensor optimized_forward(torch::Tensor x) "
                    f"{{ return {fn_name}(x); }}\n"
                )

    if "PYBIND11_MODULE" not in cuda_src:
        cuda_src = cuda_src + alias_block + "\n" + PYBIND_WRAPPER_TEMPLATE.format()
    else:
        cuda_src = cuda_src + alias_block

    return cuda_src


# ──────────────────────────────────────────────────────────────────────────────
# Compilation
# ──────────────────────────────────────────────────────────────────────────────
def compile_kernel_module(trial_id: str, cuda_src: str):
    import torch.utils.cpp_extension as cpp_ext

    prepared_src   = _prepare_cuda_source(cuda_src)
    debug_src_path = os.path.join(PENDING_DIR, f"{trial_id}_prepared.cu")
    with open(debug_src_path, "w", encoding="utf-8") as f:
        f.write(prepared_src)

    build_dir = os.path.join(PENDING_DIR, f"{trial_id}_build")
    os.makedirs(build_dir, exist_ok=True)

    module = cpp_ext.load_inline(
        name=f"kernel_{trial_id.replace('-', '_')}",
        cuda_sources=[prepared_src],
        cpp_sources=[""],
        extra_cuda_cflags=[
            f"-arch={CUDA_ARCH}",
            "--expt-relaxed-constexpr",
            "-O2",
        ],
        verbose=True,
        build_directory=build_dir,
    )
    return module, debug_src_path, build_dir


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark subprocess  — shape-aware
# ──────────────────────────────────────────────────────────────────────────────
# def _benchmark_subprocess(
#     module_build_dir: str,
#     trial_id: str,
#     cuda_src: str,               # NEW: full source so we can parse @SHAPES
#     tensor_n: int = TENSOR_N,
# ) -> dict:
#     """
#     Run correctness check + timing in a child process.

#     Strategy:
#       1. Try to parse @SHAPES from cuda_src.
#       2. If found → use declared shapes directly (no guessing).
#       3. If not found → fall back to the expanded candidate list.
#     """
#     # ── Parse shape annotation ────────────────────────────────────────────────
#     tensor_specs = parse_shapes_from_source(cuda_src)
#     if tensor_specs:
#         log_msg("INFO", f"{trial_id} | @SHAPES annotation found: {tensor_specs}")
#         declared_shapes_literal = shape_spec_to_python_literal(tensor_specs)
#         shape_mode = "declared"
#     else:
#         log_msg("WARN", f"{trial_id} | No @SHAPES annotation — using candidate list fallback.")
#         declared_shapes_literal = "null"
#         shape_mode = "candidates"

#     driver = f"""
# import sys, json, torch, importlib, glob, os
# sys.path.insert(0, {repr(module_build_dir)})

# sos = glob.glob(os.path.join({repr(module_build_dir)}, "*.so"))
# if not sos:
#     print(json.dumps({{"error": "no .so found"}})); sys.exit(1)

# name = os.path.basename(sos[0]).split(".")[0]
# mod  = importlib.import_module(name)

# device = torch.device("cuda")
# DTYPE_MAP = {{
#     "torch.float32": torch.float32,
#     "torch.float16": torch.float16,
#     "torch.int32":   torch.int32,
#     "torch.int64":   torch.int64,
# }}

# # ── Declared shapes path ──────────────────────────────────────────────────────
# declared_specs = {declared_shapes_literal}

# def make_tensors_from_specs(specs):
#     tensors = []
#     for s in specs:
#         dtype = DTYPE_MAP[s["dtype"]]
#         shape = s["shape"]
#         if dtype in (torch.int32, torch.int64):
#             t = torch.randint(0, 10, shape, dtype=dtype, device=device)
#         else:
#             t = torch.rand(shape, dtype=dtype, device=device)
#         tensors.append(t)
#     return tensors

# working_args = None

# if declared_specs is not None:
#     try:
#         tensors = make_tensors_from_specs(declared_specs)
#         mod.baseline_forward(*tensors)
#         torch.cuda.synchronize()
#         working_args = tuple(tensors)
#     except Exception as e:
#         # Annotation present but wrong — log and fall through to candidates
#         import traceback as tb
#         err_msg = tb.format_exc()
#         print(json.dumps({{"error": f"Declared @SHAPES failed: {{err_msg}}"}}))
#         sys.exit(1)

# # ── Candidate fallback (only reached if declared_specs is None) ───────────────
# if working_args is None:
#     N = 1024
#     candidates_1 = [
#         torch.rand(N, N,    dtype=torch.float32, device=device),
#         torch.rand(N, N//2, dtype=torch.float32, device=device),
#         torch.rand({tensor_n}, dtype=torch.float32, device=device),
#         torch.rand(N,       dtype=torch.float32, device=device),
#         torch.rand(4, N, N, dtype=torch.float32, device=device),
#         torch.rand(2, 4, N, N, dtype=torch.float32, device=device),
#     ]
#     candidates_2 = [
#         (torch.rand(N, N, dtype=torch.float32, device=device),
#          torch.rand(N, N, dtype=torch.float32, device=device)),
#         (torch.rand(N, N//2, dtype=torch.float32, device=device),
#          torch.rand(N//2, N, dtype=torch.float32, device=device)),
#         (torch.rand(4, N, N, dtype=torch.float32, device=device),
#          torch.rand(4, N, N, dtype=torch.float32, device=device)),
#         (torch.rand(2, 4, N, N, dtype=torch.float32, device=device),
#          torch.rand(2, 4, N, N, dtype=torch.float32, device=device)),
#         (torch.rand(N, N, dtype=torch.float32, device=device),
#          torch.rand(N, dtype=torch.float32, device=device)),
#         (torch.rand(4, N, N, dtype=torch.float32, device=device),
#          torch.rand(4, N, dtype=torch.float32, device=device)),
#         (torch.rand(N, N, dtype=torch.float32, device=device),
#          torch.rand(N, 1, dtype=torch.float32, device=device)),
#     ]

#     for x in candidates_1:
#         try:
#             mod.baseline_forward(x); torch.cuda.synchronize()
#             working_args = (x,); break
#         except: pass

#     if working_args is None:
#         for a, b in candidates_2:
#             try:
#                 mod.baseline_forward(a, b); torch.cuda.synchronize()
#                 working_args = (a, b); break
#             except: pass

#     if working_args is None:
#         print(json.dumps({{"error": "no working input shape found"}})); sys.exit(1)

# # ── Correctness + timing ──────────────────────────────────────────────────────
# ref = mod.baseline_forward(*working_args)
# opt = mod.optimized_forward(*working_args)
# torch.cuda.synchronize()

# # Use looser tolerance: large-K FP32 matmuls accumulate ~1e-3 error
# math_ok = torch.allclose(
#     ref.float().flatten(),
#     opt.float().flatten(),
#     atol=1e-3,
#     rtol=1e-3,
# )

# def time_fn(fn, args, n_warmup=5, n_bench=50):
#     for _ in range(n_warmup): fn(*args)
#     torch.cuda.synchronize()
#     s = torch.cuda.Event(enable_timing=True)
#     e = torch.cuda.Event(enable_timing=True)
#     s.record()
#     for _ in range(n_bench): fn(*args)
#     e.record(); torch.cuda.synchronize()
#     return s.elapsed_time(e) / n_bench * 1e6  # ms → ns

# base_ns = time_fn(mod.baseline_forward,  working_args)
# opt_ns  = time_fn(mod.optimized_forward, working_args)

# print(json.dumps({{"baseline_ns": base_ns, "optimized_ns": opt_ns, "math_ok": math_ok}}))
# """
#     driver_path = os.path.join(PENDING_DIR, f"{trial_id}_bench_driver.py")
#     with open(driver_path, "w") as f:
#         f.write(driver)

#     try:
#         result = run_command(
#             [sys.executable, driver_path],
#             timeout=RUN_TIMEOUT_SECONDS,
#             enforce_process_group=True,
#         )
#         cleanup_artifacts([driver_path])

#         if result.returncode != 0:
#             stderr = result.stderr.strip()
#             raise RuntimeError(
#                 f"Benchmark subprocess failed (code {result.returncode}): {stderr[:600]}"
#             )
#         return json.loads(result.stdout.strip())

#     except subprocess.TimeoutExpired:
#         cleanup_artifacts([driver_path])
#         raise RuntimeError(f"Benchmark subprocess timed out after {RUN_TIMEOUT_SECONDS}s")
#     except json.JSONDecodeError:
#         raise RuntimeError(f"Benchmark subprocess bad output: {result.stdout[:200]}")

# ── REPLACE _benchmark_subprocess ────────────────────────────────────────────
def _benchmark_subprocess(
    module_build_dir: str,
    trial_id: str,
    cuda_src: str,
    tensor_n: int = TENSOR_N,
) -> dict:
    """
    Run correctness check + timing in a child process.
    Uses declared @SHAPES if present and valid; falls back to candidates.
    """
    tensor_specs = parse_shapes_from_source(cuda_src)
    if tensor_specs:
        log_msg("INFO", f"{trial_id} | @SHAPES annotation found: {tensor_specs}")
        declared_shapes_literal = shape_spec_to_python_literal(tensor_specs)
    else:
        log_msg("WARN", f"{trial_id} | No valid @SHAPES annotation — using candidate fallback.")
        declared_shapes_literal = "null"

    # N_BENCH reduced from 50→20 to avoid timeout on slow kernels.
    # RUN_TIMEOUT handled by the outer run_command call.
    driver = f"""
import sys, json, torch, importlib, glob, os
sys.path.insert(0, {repr(module_build_dir)})

sos = glob.glob(os.path.join({repr(module_build_dir)}, "*.so"))
if not sos:
    print(json.dumps({{"error": "no .so found"}})); sys.exit(1)

name = os.path.basename(sos[0]).split(".")[0]
mod  = importlib.import_module(name)

device = torch.device("cuda")
DTYPE_MAP = {{
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.int32":   torch.int32,
    "torch.int64":   torch.int64,
}}

declared_specs = {declared_shapes_literal}

def make_tensors_from_specs(specs):
    tensors = []
    for s in specs:
        dtype = DTYPE_MAP[s["dtype"]]
        shape = s["shape"]
        if dtype in (torch.int32, torch.int64):
            t = torch.randint(0, 10, shape, dtype=dtype, device=device)
        else:
            t = torch.rand(shape, dtype=dtype, device=device)
        tensors.append(t)
    return tensors

working_args = None

if declared_specs is not None:
    try:
        tensors = make_tensors_from_specs(declared_specs)
        mod.baseline_forward(*tensors)
        torch.cuda.synchronize()
        working_args = tuple(tensors)
    except Exception as e:
        import traceback as tb
        err_msg = tb.format_exc()
        print(json.dumps({{"error": f"Declared @SHAPES failed: {{err_msg}}"}}))
        sys.exit(1)

if working_args is None:
    N = 1024
    candidates_1 = [
        torch.rand(N, N,    dtype=torch.float32, device=device),
        torch.rand(N, N//2, dtype=torch.float32, device=device),
        torch.rand({tensor_n}, dtype=torch.float32, device=device),
        torch.rand(N,       dtype=torch.float32, device=device),
        torch.rand(4, N, N, dtype=torch.float32, device=device),
        torch.rand(2, 4, N, N, dtype=torch.float32, device=device),
    ]
    candidates_2 = [
        (torch.rand(N, N, dtype=torch.float32, device=device),
         torch.rand(N, N, dtype=torch.float32, device=device)),
        (torch.rand(N, N//2, dtype=torch.float32, device=device),
         torch.rand(N//2, N, dtype=torch.float32, device=device)),
        (torch.rand(4, N, N, dtype=torch.float32, device=device),
         torch.rand(4, N, N, dtype=torch.float32, device=device)),
        (torch.rand(2, 4, N, N, dtype=torch.float32, device=device),
         torch.rand(2, 4, N, N, dtype=torch.float32, device=device)),
        (torch.rand(N, N, dtype=torch.float32, device=device),
         torch.rand(N, dtype=torch.float32, device=device)),
        (torch.rand(4, N, N, dtype=torch.float32, device=device),
         torch.rand(4, N, dtype=torch.float32, device=device)),
        (torch.rand(N, N, dtype=torch.float32, device=device),
         torch.rand(N, 1, dtype=torch.float32, device=device)),
    ]
    for x in candidates_1:
        try:
            mod.baseline_forward(x); torch.cuda.synchronize()
            working_args = (x,); break
        except: pass
    if working_args is None:
        for a, b in candidates_2:
            try:
                mod.baseline_forward(a, b); torch.cuda.synchronize()
                working_args = (a, b); break
            except: pass
    if working_args is None:
        print(json.dumps({{"error": "no working input shape found"}})); sys.exit(1)

ref = mod.baseline_forward(*working_args)
opt = mod.optimized_forward(*working_args)
torch.cuda.synchronize()

# Capture mismatch details for RL signal
max_diff = float((ref.float() - opt.float()).abs().max().item())
math_ok  = max_diff <= 1e-3

def time_fn(fn, args, n_warmup=5, n_bench=20):  # n_bench reduced 50→20
    for _ in range(n_warmup): fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_bench): fn(*args)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n_bench * 1e6

base_ns = time_fn(mod.baseline_forward,  working_args)
opt_ns  = time_fn(mod.optimized_forward, working_args)

print(json.dumps({{
    "baseline_ns":  base_ns,
    "optimized_ns": opt_ns,
    "math_ok":      math_ok,
    "max_diff":     max_diff,
}}))
"""
    driver_path = os.path.join(PENDING_DIR, f"{trial_id}_bench_driver.py")
    with open(driver_path, "w") as f:
        f.write(driver)

    try:
        result = run_command(
            [sys.executable, driver_path],
            timeout=60.0,           # increased from 30s → 60s
            enforce_process_group=True,
        )
        cleanup_artifacts([driver_path])
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"Benchmark subprocess failed (code {result.returncode}): {stderr[:600]}"
            )
        return json.loads(result.stdout.strip())
    except subprocess.TimeoutExpired:
        cleanup_artifacts([driver_path])
        raise RuntimeError("Benchmark subprocess timed out after 60s")
    except json.JSONDecodeError:
        raise RuntimeError(f"Benchmark subprocess bad output: {result.stdout[:200]}")
    
# ──────────────────────────────────────────────────────────────────────────────
# Timing helper  (kept for direct in-process use if needed)
# ──────────────────────────────────────────────────────────────────────────────
def _cuda_event_time_ms(fn, *args, n_warmup=N_WARMUP, n_bench=N_BENCH) -> float:
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_bench):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_bench


# ──────────────────────────────────────────────────────────────────────────────
# Nsight Compute profiling  (unchanged from v3)
# ──────────────────────────────────────────────────────────────────────────────
def profile_with_ncu(module_build_dir: str, trial_id: str) -> "dict | None":
    driver_src = f"""\
import torch, sys
sys.path.insert(0, {repr(module_build_dir)})
import importlib, os, glob

sos = glob.glob(os.path.join({repr(module_build_dir)}, "*.so"))
if not sos:
    print("NO_SO_FOUND"); sys.exit(1)

name = os.path.basename(sos[0]).split(".")[0]
mod  = importlib.import_module(name)

x = torch.rand({TENSOR_N}, dtype=torch.float32, device="cuda")
mod.baseline_forward(x)
mod.optimized_forward(x)
torch.cuda.synchronize()
"""
    driver_path = os.path.join(PENDING_DIR, f"{trial_id}_ncu_driver.py")
    with open(driver_path, "w", encoding="utf-8") as f:
        f.write(driver_src)

    ncu_cmd = [
        "ncu",
        "--launch-count", "2",
        "--csv",
        "--metrics", ",".join(OPTIMIZED_METRICS),
        sys.executable, driver_path,
    ]
    try:
        ncu_res = run_command(ncu_cmd, timeout=PROFILE_TIMEOUT_SECONDS, enforce_process_group=True)
    finally:
        cleanup_artifacts([driver_path])

    if ncu_res.returncode != 0:
        log_msg("WARN", f"ncu returned non-zero for {trial_id}: {ncu_res.stderr[:400]}")
        return None

    try:
        df = parse_ncu_csv_to_dataframe(ncu_res.stdout)
        baseline_kernel, optimized_kernel = choose_kernel_names(df)
        occ  = extract_metric_for_kernel_name(df, optimized_kernel, "sm__warps_active.avg.pct_of_peak_sustained_active")
        comp = extract_metric_for_kernel_name(df, optimized_kernel, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
        mem  = extract_metric_for_kernel_name(df, optimized_kernel, "dram__throughput.avg.pct_of_peak_sustained_elapsed")
        return {"occ": occ, "comp": comp, "mem": mem}
    except Exception as e:
        log_msg("WARN", f"ncu CSV parse failed for {trial_id}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Finalize helper
# ──────────────────────────────────────────────────────────────────────────────
def finalize_trial(
    trial_id, result_path, status, reward, error_trace, m,
    cleanup_paths, keep_artifacts=False
) -> None:
    payload = build_result_payload(
        trial_id, status, reward, error_trace,
        m["base_t"], m["opt_t"], m["occ"], m["comp"], m["mem"]
    )
    try:
        atomic_write_json(result_path, payload, trial_id)
        log_msg("INFO", f"Finished {trial_id} | Status: {status} | Reward: {reward:.4f}")
    finally:
        if not (keep_artifacts and KEEP_FAILED_ARTIFACTS):
            cleanup_artifacts(cleanup_paths)


# ──────────────────────────────────────────────────────────────────────────────
# Trial pipeline  — one change: pass cuda_src to _benchmark_subprocess
# ──────────────────────────────────────────────────────────────────────────────
def process_trial(trial_id: str) -> None:
    kernel_cu   = os.path.join(PENDING_DIR,   f"{trial_id}_kernel.cu")
    result_path = os.path.join(COMPLETED_DIR, f"{trial_id}_results.json")

    if os.path.exists(result_path):
        log_msg("INFO", f"Skipping already-completed trial: {trial_id}")
        cleanup_artifacts([kernel_cu])
        return

    if not os.path.exists(kernel_cu):
        return

    if not is_file_stable(kernel_cu):
        log_msg("INFO", f"Deferring trial {trial_id}: input file still being written.")
        return

    lock_path = acquire_trial_lock(trial_id)
    if lock_path is None:
        return

    log_msg("INFO", f"Processing trial: {trial_id}")

    metrics     = {"base_t": None, "opt_t": None, "occ": None, "comp": None, "mem": None}
    status      = "SUCCESS"
    reward      = -1.0
    error_trace = None
    debug_src_path = None
    build_dir      = None

    try:
        with open(kernel_cu, "r", encoding="utf-8") as f:
            cuda_src = f.read()

        # Step 1: Compile
        log_msg("INFO", f"{trial_id} | Compiling via load_inline …")
        try:
            module, debug_src_path, build_dir = compile_kernel_module(trial_id, cuda_src)
            log_msg("INFO", f"{trial_id} | Compilation SUCCESS")
        except Exception as compile_exc:
            error_trace = compact_text(str(compile_exc))
            log_msg("WARN", f"{trial_id} | COMPILATION_ERROR: {error_trace[:200]}")
            return finalize_trial(
                trial_id, result_path, "COMPILATION_ERROR", -1.0, error_trace, metrics,
                [kernel_cu, debug_src_path] if debug_src_path else [kernel_cu],
                keep_artifacts=True,
            )

        # Step 2: Benchmark — pass cuda_src so subprocess can parse @SHAPES
        log_msg("INFO", f"{trial_id} | Benchmarking …")
        try:
            bench_result = _benchmark_subprocess(build_dir, trial_id, cuda_src)
            if "error" in bench_result:
                raise RuntimeError(bench_result["error"])
            baseline_ns  = bench_result["baseline_ns"]
            optimized_ns = bench_result["optimized_ns"]
            # ── In process_trial(), replace the math_ok check block ───────────────
            math_ok  = bench_result["math_ok"]
            max_diff = bench_result.get("max_diff", None)

            if not math_ok:
                error_trace = (
                    f"torch.allclose failed: max_diff={max_diff:.6f} "
                    f"(threshold=1e-3). Optimized output differs from baseline."
                )
                return finalize_trial(
                    trial_id, result_path, "WRONG_MATH", 0.0, error_trace, metrics,
                    [kernel_cu],
                )
        except Exception as bench_exc:
            error_trace = compact_text(traceback.format_exc())
            return finalize_trial(
                trial_id, result_path, "RUNTIME_ERROR", -1.0, error_trace, metrics,
                [kernel_cu], keep_artifacts=True,
            )

        if not math_ok:
            error_trace = "torch.allclose failed: optimized output differs from baseline."
            return finalize_trial(
                trial_id, result_path, "WRONG_MATH", 0.0, error_trace, metrics,
                [kernel_cu],
            )

        metrics["base_t"] = baseline_ns
        metrics["opt_t"]  = optimized_ns

        if baseline_ns <= 0 or optimized_ns <= 0:
            error_trace = f"Invalid timing: baseline={baseline_ns} ns, optimized={optimized_ns} ns"
            return finalize_trial(
                trial_id, result_path, "SYSTEM_ERROR", -1.0, error_trace, metrics,
                [kernel_cu], keep_artifacts=True,
            )

        reward = baseline_ns / optimized_ns
        log_msg("INFO",
            f"{trial_id} | baseline={baseline_ns/1e6:.3f} ms  "
            f"optimized={optimized_ns/1e6:.3f} ms  reward={reward:.4f}"
        )

        # Step 3: ncu (best-effort)
        if build_dir:
            log_msg("INFO", f"{trial_id} | Running ncu …")
            ncu_metrics = profile_with_ncu(build_dir, trial_id)
            if ncu_metrics:
                metrics["occ"]  = ncu_metrics["occ"]
                metrics["comp"] = ncu_metrics["comp"]
                metrics["mem"]  = ncu_metrics["mem"]
            else:
                log_msg("WARN", f"{trial_id} | ncu profiling failed — timing still valid.")

        return finalize_trial(
            trial_id, result_path, "SUCCESS", reward, None, metrics, [kernel_cu]
        )

    except Exception as exc:
        error_trace = compact_text(traceback.format_exc())
        log_msg("CRITICAL", f"Trial {trial_id} unhandled exception: {exc}")
        finalize_trial(
            trial_id, result_path, "SYSTEM_ERROR", -1.0, error_trace, metrics,
            [kernel_cu], keep_artifacts=True,
        )
    finally:
        release_trial_lock(lock_path)


# ──────────────────────────────────────────────────────────────────────────────
# Watcher main loop  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ensure_directories()
    log_msg("INFO", "Script 2 v4 evaluator started. Monitoring pending_kernels/ …")
    while True:
        try:
            trial_ids = list_trial_ids()
            if not trial_ids:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            for trial_id in trial_ids:
                process_trial(trial_id)
            time.sleep(POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            log_msg("INFO", "Exiting evaluator.")
            break
        except Exception as exc:
            log_msg("CRITICAL", f"Watcher loop error: {exc}")
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()