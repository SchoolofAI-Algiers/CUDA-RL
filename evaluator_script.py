#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import torch
from torch.utils.cpp_extension import load as cpp_load


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))

BASELINE_MODE         = os.environ.get("BASELINE_MODE",       "cuda")
REWARD_MODE           = os.environ.get("REWARD_MODE",         "speedup")
ATOL                  = _env_float("EVAL_ATOL",                1e-2)
RTOL                  = _env_float("EVAL_RTOL",                1e-2)
TIMING_WARMUP         = _env_int("EVAL_TIMING_WARMUP",         3)
TIMING_RUNS           = _env_int("EVAL_TIMING_RUNS",           10)
POLL_INTERVAL_SECONDS = _env_float("EVAL_POLL_INTERVAL",       1.0)
FILE_STABLE_WAIT      = _env_float("EVAL_FILE_STABLE_WAIT",    0.3)
PENDING_DIR           = os.environ.get("PENDING_DIR",          "pending_kernels")
COMPLETED_DIR         = os.environ.get("COMPLETED_DIR",        "completed_results")


def _find_compiler(candidates: list[str]) -> str | None:
    dirs = [
        os.path.join(os.environ.get("CONDA_PREFIX", "/opt/conda"), "bin"),
        "/opt/conda/bin", "/usr/bin", "/usr/local/bin",
    ]
    for d in dirs:
        for name in candidates:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return None

_CC  = _find_compiler(["x86_64-conda-linux-gnu-cc",  "gcc",  "cc"])
_CXX = _find_compiler(["x86_64-conda-linux-gnu-c++", "g++", "c++"])
if _CC:  os.environ["CC"]  = _CC
if _CXX: os.environ["CXX"] = _CXX
os.environ["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + os.environ.get("LD_LIBRARY_PATH", "")


def log_msg(level: str, msg: str) -> None:
    ts     = time.strftime("%Y-%m-%d %H:%M:%S")
    target = sys.stderr if level == "CRITICAL" else sys.stdout
    print(f"[{ts}] [{level}] {msg}", file=target, flush=True)


def short_hash(s: str, length: int = 8) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:length]


def ensure_directories() -> None:
    os.makedirs(PENDING_DIR,   exist_ok=True)
    os.makedirs(COMPLETED_DIR, exist_ok=True)


def cleanup_stale_locks() -> None:
    for name in os.listdir(PENDING_DIR):
        if name.startswith(".") and name.endswith(".lock"):
            lock_path = os.path.join(PENDING_DIR, name)
            try:
                os.remove(lock_path)
                log_msg("WARNING", f"Removed stale lock: {lock_path}")
            except FileNotFoundError:
                pass
            except Exception as exc:
                log_msg("WARNING", f"Could not remove stale lock {lock_path}: {exc}")


def list_trial_ids() -> list[str]:
    ids = [
        n[: -len("_candidate.cu")]
        for n in os.listdir(PENDING_DIR)
        if n.endswith("_candidate.cu")
    ]
    ids = [
        tid for tid in ids
        if os.path.exists(os.path.join(PENDING_DIR, f"{tid}_reference.cu"))
    ]
    ids.sort()
    return ids


def is_file_stable(path: str, wait: float = FILE_STABLE_WAIT) -> bool:
    if not os.path.exists(path):
        return False
    try:
        before = os.stat(path)
        time.sleep(wait)
        after  = os.stat(path)
    except FileNotFoundError:
        return False
    return (before.st_size == after.st_size
            and before.st_mtime_ns == after.st_mtime_ns)


def acquire_trial_lock(trial_id: str) -> str | None:
    lock_path = os.path.join(PENDING_DIR, f".{trial_id}.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return lock_path
    except FileExistsError:
        return None


def release_trial_lock(lock_path: str | None) -> None:
    if lock_path and os.path.exists(lock_path):
        try:
            os.remove(lock_path)
        except Exception:
            pass


def cleanup_artifacts(paths: list[str]) -> None:
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


def atomic_write_json(result_path: str, payload: dict, trial_id: str) -> None:
    temp_path = os.path.join(COMPLETED_DIR, f"temp_{trial_id}.json")
    with open(temp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.rename(temp_path, result_path)


def build_result_payload(
    trial_id:     str,
    status:       str,
    reward:       float | None,
    error_trace:  str | None,
    ref_time:     float | None,
    cuda_time:    float | None,
    op_name:      str = "unknown",
    extra_fields: dict | None = None,
) -> dict:
    payload: dict[str, Any] = {
        "trial_id":    trial_id,
        "op_name":     op_name,
        "status":      status,
        "reward":      float(reward) if reward is not None else -1.0,
        "error_trace": error_trace,
        "metrics": {
            "ref_time_ms":    ref_time,
            "cuda_time_ms":   cuda_time,
            "ref_stddev_ms":  None,
            "cuda_stddev_ms": None,
            "allclose":       False,
            "max_abs_diff":   float("inf"),
        },
    }
    if extra_fields:
        for k, v in extra_fields.items():
            if k in payload["metrics"]:
                payload["metrics"][k] = v
            else:
                payload[k] = v
    return payload


def fix_mismatched_kernel_name(code: str) -> tuple[str, str | None, str | None]:
    defined = re.findall(r'__global__\s+\w+\s+(\w+)\s*\(', code)
    called  = re.findall(r'(\w+)\s*<<<', code)
    if len(defined) == 1 and len(called) == 1 and defined[0] != called[0]:
        old, new = called[0], defined[0]
        fixed    = re.sub(rf'\b{re.escape(old)}\s*<<<', f'{new}<<<', code)
        return fixed, old, new
    return code, None, None


def get_undefined_kernels(code: str) -> set[str]:
    defined = set(re.findall(r'__global__\s+\w+\s+(\w+)\s*\(', code))
    called  = set(re.findall(r'(\w+)\s*<<<', code))
    return called - defined


def compile_extension(cuda_code: str, ext_name: str):
    build_dir = tempfile.mkdtemp(prefix=f"build_{ext_name}_")
    src_path  = os.path.join(build_dir, f"{ext_name}.cu")
    with open(src_path, "w") as f:
        f.write(cuda_code)
    old_torch_ext = os.environ.get("TORCH_EXTENSIONS_DIR")
    os.environ["TORCH_EXTENSIONS_DIR"] = build_dir
    try:
        mod = cpp_load(
            name=ext_name,
            sources=[src_path],
            verbose=False,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-Wno-deprecated-gpu-targets"],
        )
    finally:
        if old_torch_ext is None:
            os.environ.pop("TORCH_EXTENSIONS_DIR", None)
        else:
            os.environ["TORCH_EXTENSIONS_DIR"] = old_torch_ext
        shutil.rmtree(build_dir, ignore_errors=True)
    return mod


def get_real_nvcc_error(cuda_code: str) -> str:
    import subprocess, sysconfig
    nvcc = shutil.which("nvcc") or "/opt/conda/bin/nvcc"
    torch_dir: str | None = None
    try:
        import torch as _torch
        torch_dir = os.path.dirname(_torch.__file__)
    except Exception:
        pass

    build_dir = tempfile.mkdtemp(prefix="probe_")
    cu_path   = os.path.join(build_dir, "probe.cu")
    with open(cu_path, "w") as f:
        f.write(cuda_code)
    try:
        cmd = [
            nvcc,
            "-O2", "--expt-relaxed-constexpr",
            "--compiler-options", "-fPIC", "-std=c++17",
            "-c", cu_path, "-o", os.path.join(build_dir, "probe.o"),
        ]
        if torch_dir:
            cmd += [
                "-isystem", os.path.join(torch_dir, "include"),
                "-isystem", os.path.join(torch_dir, "include",
                                         "torch", "csrc", "api", "include"),
            ]
        py_inc = sysconfig.get_path("include")
        if py_inc:
            cmd += ["-isystem", py_inc]
        probe = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if probe.returncode == 0:
            return ""
        return probe.stderr or probe.stdout or "(no output)"
    except Exception as exc:
        return f"(probe failed: {exc})"
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


def compile_both(
    cand_code: str, cand_ext: str,
    ref_code:  str, ref_ext:  str,
) -> tuple[Any, Any, str | None]:
    results: dict[str, Any]       = {}
    errors:  dict[str, Exception] = {}

    def _compile(code: str, ext: str, key: str) -> None:
        try:
            results[key] = compile_extension(code, ext)
        except Exception as exc:
            errors[key] = exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_compile, cand_code, cand_ext, "candidate"): "candidate",
            pool.submit(_compile, ref_code,  ref_ext,  "reference"):  "reference",
        }
        for fut in as_completed(futures):
            fut.result()

    if errors:
        parts: list[str] = []
        for side, exc in errors.items():
            code     = cand_code if side == "candidate" else ref_code
            nvcc_out = get_real_nvcc_error(code)
            parts.append(
                f"[{side.upper()}] cpp_load: {exc}"
                + (f"\nnvcc:\n{nvcc_out}" if nvcc_out else "")
            )
        return None, None, "\n\n".join(parts)

    return results["candidate"], results["reference"], None


def find_kernel_callable(mod):
    candidates = [x for x in dir(mod) if not x.startswith("__")]
    for preferred in ("forward", "matmul", "kernel", "run", "compute"):
        if hasattr(mod, preferred) and callable(getattr(mod, preferred)):
            return getattr(mod, preferred)
    for name in candidates:
        attr = getattr(mod, name)
        if callable(attr):
            log_msg("INFO", f"Using fallback callable: {name}")
            return attr
    raise AttributeError(f"No callable found. Exports: {candidates}")


def _make_inputs_for_op(op_name: str) -> tuple:
    op = op_name.lower()

    if "masked_cumsum" in op:
        x    = torch.randn(256, 1024, device="cuda")
        mask = torch.ones(256, 1024, dtype=torch.bool, device="cuda")
        return (x, mask, 1)

    if "cumsum" in op or "cumprod" in op or "scan" in op:
        return (torch.randn(256, 1024, device="cuda"),)

    if "diagonal" in op and any(k in op for k in ["matmul", "gemm", "mat_mul"]):
        return (
            torch.randn(512,      device="cuda"),
            torch.randn(512, 512, device="cuda"),
        )

    if any(k in op for k in ["matmul", "gemm", "matrix_mul", "mat_mul", "linear"]):
        return (
            torch.randn(512, 512, device="cuda"),
            torch.randn(512, 512, device="cuda"),
        )

    if "bmm" in op or ("batch" in op and "matmul" in op):
        return (
            torch.randn(8, 256, 256, device="cuda"),
            torch.randn(8, 256, 256, device="cuda"),
        )

    if "rmsnorm" in op or "rms_norm" in op:
        return (torch.randn(4, 256, 768, device="cuda"), 1e-6)

    if any(k in op for k in ["layernorm", "layer_norm"]):
        B, T, C = 4, 256, 768
        return (
            torch.randn(B, T, C, device="cuda"),
            torch.ones(C,        device="cuda"),
            torch.zeros(C,       device="cuda"),
        )

    if any(k in op for k in ["groupnorm", "group_norm", "instancenorm"]):
        return (torch.randn(4, 32, 64, 64, device="cuda"),)

    if "softmax" in op:
        return (torch.randn(128, 1024, device="cuda"),)

    if any(k in op for k in ["cross_entropy", "nll", "loss"]):
        return (
            torch.randn(256, 1024, device="cuda"),
            torch.randint(0, 1024, (256,), device="cuda"),
        )

    if any(k in op for k in ["gelu", "silu", "relu", "sigmoid", "tanh", "activation"]):
        return (torch.randn(512, 1024, device="cuda"),)

    if "dropout" in op:
        return (torch.randn(512, 1024, device="cuda"),)

    if any(k in op for k in ["add", "mul", "div", "elementwise",
                              "element_wise", "hadamard"]):
        return (
            torch.randn(512, 512, device="cuda"),
            torch.randn(512, 512, device="cuda"),
        )

    if any(k in op for k in ["conv", "convolution"]):
        return (
            torch.randn(4, 64, 32, 32, device="cuda"),
            torch.randn(64, 64, 3, 3,  device="cuda"),
        )

    if any(k in op for k in ["attention", "attn", "scaled_dot"]):
        B, H, T, D = 2, 8, 128, 64
        return (
            torch.randn(B, H, T, D, device="cuda"),
            torch.randn(B, H, T, D, device="cuda"),
            torch.randn(B, H, T, D, device="cuda"),
        )

    if any(k in op for k in ["rope", "rotary", "positional"]):
        return (torch.randn(4, 128, 8, 64, device="cuda"),)

    if any(k in op for k in ["embedding", "gather", "scatter"]):
        return (
            torch.randn(10000, 256, device="cuda"),
            torch.randint(0, 10000, (32, 64), device="cuda"),
        )

    if any(k in op for k in ["sort", "topk", "top_k"]):
        return (torch.randn(256, 1024, device="cuda"),)

    if any(k in op for k in ["reduce", "reduction", "sum", "mean", "max"]):
        return (torch.randn(256, 1024, device="cuda"),)

    if any(k in op for k in ["transpose", "permute"]):
        return (torch.randn(256, 512, device="cuda"),)

    log_msg("WARNING", f"No input spec matched op='{op_name}' — using generic 2-D fallback")
    return (torch.randn(256, 512, device="cuda"),)


def _try_inputs(
    cand_fn, ref_fn, inputs: tuple
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        ref_inputs  = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in inputs)
        cand_inputs = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in inputs)
        ref_out     = ref_fn(*ref_inputs)
        cand_out    = cand_fn(*cand_inputs)

    def _resolve(out, fallback_inputs):
        # In-place kernels return None; the result lives in the first input tensor.
        if out is None:
            first = next((t for t in fallback_inputs if isinstance(t, torch.Tensor)), None)
            if first is None:
                raise ValueError("Kernel returned None and no input tensor found to inspect.")
            return first
        return out[0] if isinstance(out, (list, tuple)) else out

    return _resolve(ref_out, ref_inputs), _resolve(cand_out, cand_inputs)


def time_ms(
    fn, inputs: tuple,
    warmup: int = TIMING_WARMUP,
    runs:   int = TIMING_RUNS,
) -> tuple[float, float]:
    for _ in range(warmup):
        fn(*inputs)
    torch.cuda.synchronize()  # flush residual GPU work before opening events

    start  = torch.cuda.Event(enable_timing=True)
    end    = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(runs):
        start.record()
        fn(*inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean   = sum(times) / len(times)
    stddev = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, stddev


def evaluate_cuda_mode(
    trial_id:       str,
    candidate_code: str,
    reference_code: str,
    op_name:        str = "unknown",
) -> dict:
    cand_code, old_c, new_c = fix_mismatched_kernel_name(candidate_code)
    ref_code,  old_r, new_r = fix_mismatched_kernel_name(reference_code)
    if old_c:
        log_msg("INFO", f"Fixed candidate kernel name: {old_c} -> {new_c}")
    if old_r:
        log_msg("INFO", f"Fixed reference  kernel name: {old_r} -> {new_r}")

    undefined = get_undefined_kernels(cand_code) | get_undefined_kernels(ref_code)
    if undefined:
        return build_result_payload(
            trial_id, "SKIP_MISSING_KERNEL", -1.0,
            f"Undefined kernels: {undefined}", None, None, op_name=op_name,
        )

    h = short_hash(trial_id)
    cand_mod, ref_mod, compile_err = compile_both(
        cand_code, f"k_{h}_c", ref_code, f"k_{h}_r"
    )
    if compile_err is not None:
        status = (
            "CANDIDATE_COMPILE_ERROR"
            if "[CANDIDATE]" in compile_err and "[REFERENCE]" not in compile_err
            else "REFERENCE_COMPILE_ERROR"
            if "[REFERENCE]" in compile_err and "[CANDIDATE]" not in compile_err
            else "COMPILATION_ERROR"
        )
        return build_result_payload(
            trial_id, status, -1.0, compile_err, None, None, op_name=op_name,
        )

    try:
        cand_fn = find_kernel_callable(cand_mod)
        ref_fn  = find_kernel_callable(ref_mod)
    except AttributeError as exc:
        return build_result_payload(
            trial_id, "COMPILATION_ERROR", -1.0, str(exc),
            None, None, op_name=op_name,
        )

    inputs = _make_inputs_for_op(op_name)
    log_msg(
        "INFO",
        f"op='{op_name}' → inputs: "
        f"{[tuple(t.shape) if hasattr(t, 'shape') else t for t in inputs]}",
    )

    try:
        ref_out, cand_out = _try_inputs(cand_fn, ref_fn, inputs)
    except Exception as exc:
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
        return build_result_payload(
            trial_id, "RUNTIME_ERROR", -1.0,
            f"Kernel exec: {exc}", None, None, op_name=op_name,
        )

    max_diff = (cand_out.float() - ref_out.float()).abs().max().item()
    allclose  = torch.allclose(cand_out.float(), ref_out.float(), atol=ATOL, rtol=RTOL)
    if not allclose:
        return build_result_payload(
            trial_id, "WRONG_MATH", 0.0,
            f"max_diff={max_diff:.6f}", None, None, op_name=op_name,
            extra_fields={"allclose": False, "max_abs_diff": max_diff},
        )

    try:
        cand_ms, cand_std = time_ms(cand_fn, inputs)
        ref_ms,  ref_std  = time_ms(ref_fn,  inputs)
        speedup = ref_ms / cand_ms if cand_ms > 0 else -1.0
    except Exception as exc:
        return build_result_payload(
            trial_id, "SYSTEM_ERROR", -1.0,
            f"Timing: {exc}", None, None, op_name=op_name,
        )

    reward = speedup if REWARD_MODE == "speedup" else 1.0
    return build_result_payload(
        trial_id, "SUCCESS", reward, None, ref_ms, cand_ms, op_name=op_name,
        extra_fields={
            "speedup":        speedup,
            "allclose":       True,
            "max_abs_diff":   max_diff,
            "ref_stddev_ms":  ref_std,
            "cuda_stddev_ms": cand_std,
        },
    )


def process_trial(trial_id: str) -> None:
    cand_path   = os.path.join(PENDING_DIR,   f"{trial_id}_candidate.cu")
    ref_path    = os.path.join(PENDING_DIR,   f"{trial_id}_reference.cu")
    meta_path   = os.path.join(PENDING_DIR,   f"{trial_id}_meta.json")
    result_path = os.path.join(COMPLETED_DIR, f"{trial_id}_results.json")

    if os.path.exists(result_path):
        cleanup_artifacts([cand_path, ref_path, meta_path])
        return

    if not is_file_stable(cand_path) or not is_file_stable(ref_path):
        return

    with open(cand_path, "r", errors="replace") as f:
        cand_code = f.read()
    with open(ref_path, "r", errors="replace") as f:
        ref_code  = f.read()

    op_name = "unknown"
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                op_name = json.load(f).get("op_name", "unknown")
        except Exception as exc:
            log_msg("WARNING", f"Could not read meta for {trial_id}: {exc}")

    result = evaluate_cuda_mode(trial_id, cand_code, ref_code, op_name=op_name)
    atomic_write_json(result_path, result, trial_id)
    cleanup_artifacts([cand_path, ref_path, meta_path])
    log_msg(
        "INFO",
        f"Completed {trial_id} [{op_name}] → {result['status']}  "
        f"reward={result['reward']:.4f}",
    )


def print_summary() -> None:
    results: list[dict] = []
    for fn in os.listdir(COMPLETED_DIR):
        if fn.endswith("_results.json") and not fn.startswith("temp_"):
            try:
                with open(os.path.join(COMPLETED_DIR, fn)) as f:
                    results.append(json.load(f))
            except Exception as exc:
                log_msg("WARNING", f"Could not read result file {fn}: {exc}")

    if not results:
        log_msg("INFO", "No results to summarise.")
        return

    from collections import Counter
    statuses = Counter(r.get("status", "?") for r in results)

    print("\n" + "=" * 108)
    print(f"REWARD_MODE = {REWARD_MODE}")
    print(
        f"{'Trial ID':<30} {'Op':<22} {'Status':<28} "
        f"{'Speedup':>8} {'Cand ms':>9} {'Ref ms':>9} {'Reward':>8}"
    )
    print("-" * 108)
    for r in sorted(results, key=lambda x: -(x.get("reward") or -1)):
        sid     = r.get("trial_id", "?")[:28]
        op      = (r.get("op_name") or r.get("metrics", {}).get("op_name") or "?")[:20]
        metrics = r.get("metrics", {})
        spd     = r.get("speedup")
        cand_ms = metrics.get("cuda_time_ms")
        ref_ms  = metrics.get("ref_time_ms")
        print(
            f"{sid:<30} {op:<22} {r.get('status', '?'):<28} "
            f"{f'{spd:.3f}x' if spd is not None else '—':>8} "
            f"{f'{cand_ms:.2f}' if cand_ms is not None else '—':>9} "
            f"{f'{ref_ms:.2f}'  if ref_ms  is not None else '—':>9} "
            f"{r.get('reward', -1.0):>8.3f}"
        )

    print("=" * 108)
    print("Status breakdown:", dict(statuses))
    successes = [r for r in results if r.get("status") == "SUCCESS"]
    if successes:
        avg = sum(r.get("reward", 0.0) for r in successes) / len(successes)
        print(f"SUCCESS: {len(successes)}  avg reward: {avg:.3f}")


def main() -> None:
    ensure_directories()
    cleanup_stale_locks()
    log_msg(
        "INFO",
        f"Evaluator started (BASELINE_MODE={BASELINE_MODE}, "
        f"REWARD_MODE={REWARD_MODE}). Watching {PENDING_DIR}/ ...",
    )

    while True:
        try:
            pending = list_trial_ids()
            if pending:
                for tid in pending:
                    lock = acquire_trial_lock(tid)
                    if not lock:
                        continue
                    try:
                        process_trial(tid)
                    finally:
                        release_trial_lock(lock)
            else:
                time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            log_msg("INFO", "Evaluator stopped by user.")
            break

        except Exception as exc:
            log_msg("CRITICAL", f"Loop error: {exc}\n{traceback.format_exc()}")
            is_cuda_fatal = (
                "CUDA error" in str(exc)
                or "misaligned" in str(exc)
                or "AcceleratorError" in type(exc).__name__
            )
            if is_cuda_fatal:
                log_msg("CRITICAL", "CUDA context poisoned — marking pending trials as failed and restarting.")
                try:
                    for tid in list_trial_ids():
                        result_path = os.path.join(COMPLETED_DIR, f"{tid}_results.json")
                        if not os.path.exists(result_path):
                            payload = build_result_payload(
                                tid, "RUNTIME_ERROR", -1.0,
                                f"CUDA context poisoned: {exc}",
                                None, None, op_name="unknown",
                            )
                            atomic_write_json(result_path, payload, tid)
                            cleanup_artifacts([
                                os.path.join(PENDING_DIR, f"{tid}_candidate.cu"),
                                os.path.join(PENDING_DIR, f"{tid}_reference.cu"),
                                os.path.join(PENDING_DIR, f"{tid}_meta.json"),
                            ])
                            log_msg("INFO", f"Marked {tid} as RUNTIME_ERROR (CUDA context failure)")
                except Exception as cleanup_err:
                    log_msg("CRITICAL", f"Cleanup failed: {cleanup_err}")

                for name in os.listdir(PENDING_DIR):
                    if name.startswith(".") and name.endswith(".lock"):
                        try:
                            os.remove(os.path.join(PENDING_DIR, name))
                        except Exception:
                            pass

                log_msg("INFO", "Re-executing evaluator process...")
                os.execv(sys.executable, [sys.executable] + sys.argv)

            time.sleep(POLL_INTERVAL_SECONDS)

    print_summary()


if __name__ == "__main__":
    main()
