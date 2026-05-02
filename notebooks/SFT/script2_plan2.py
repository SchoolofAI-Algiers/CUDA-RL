#!/usr/bin/env python3
# =============================================================================
# Evaluator — CUDA baseline mode
# Compares candidate kernel against reference kernel submitted by the notebook.
# Fixes vs original:
#   1. Input generation is op-name-aware (not hardcoded matmul for everything)
#   2. metrics dict includes allclose + max_abs_diff so notebook correctness works
#   3. ext_name is derived from a short hash, never from the full trial_id string
#   4. compile_extension ext_name is always unique between candidate and reference
# =============================================================================

import os, sys, re, json, shutil, tempfile, time, traceback, hashlib
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load as cpp_load

# ========== Configuration ==========
BASELINE_MODE       = "cuda"
REWARD_MODE         = "speedup"   # "correctness_only" or "speedup"
ATOL                = 1e-2
RTOL                = 1e-2
TIMING_WARMUP       = 3
TIMING_RUNS         = 10
PENDING_DIR         = "pending_kernels"
COMPLETED_DIR       = "completed_results"
POLL_INTERVAL_SECONDS = 1.0
FILE_STABLE_WAIT    = 0.3

# ========== Force Conda toolchain ==========
def find_compiler(candidates):
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

CC  = find_compiler(["x86_64-conda-linux-gnu-cc",  "gcc",  "cc"])
CXX = find_compiler(["x86_64-conda-linux-gnu-c++", "g++", "c++"])
if CC:  os.environ["CC"]  = CC
if CXX: os.environ["CXX"] = CXX
os.environ["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

# ========== Logging ==========
def log_msg(level, msg):
    ts     = time.strftime("%Y-%m-%d %H:%M:%S")
    target = sys.stderr if level == "CRITICAL" else sys.stdout
    print(f"[{ts}] [{level}] {msg}", file=target, flush=True)

# ========== Directory / file helpers ==========
def ensure_directories():
    os.makedirs(PENDING_DIR,   exist_ok=True)
    os.makedirs(COMPLETED_DIR, exist_ok=True)

def list_trial_ids():
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

def is_file_stable(path, wait=FILE_STABLE_WAIT):
    if not os.path.exists(path):
        return False
    try:
        before = os.stat(path)
        time.sleep(wait)
        after  = os.stat(path)
    except FileNotFoundError:
        return False
    return before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns

def acquire_trial_lock(trial_id):
    lock_path = os.path.join(PENDING_DIR, f".{trial_id}.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return lock_path
    except FileExistsError:
        return None

def release_trial_lock(lock_path):
    if lock_path and os.path.exists(lock_path):
        try:
            os.remove(lock_path)
        except Exception:
            pass

def cleanup_artifacts(paths):
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

def atomic_write_json(result_path, payload, trial_id):
    temp_path = os.path.join(COMPLETED_DIR, f"temp_{trial_id}.json")
    with open(temp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.rename(temp_path, result_path)

def build_result_payload(trial_id, status, reward, error_trace,
                         ref_time, cuda_time, extra_fields=None):
    """
    Base payload. extra_fields may include speedup, allclose, max_abs_diff.
    The notebook's evaluate_kernel_pair reads metrics.allclose and
    metrics.max_abs_diff — always include them here.
    """
    payload = {
        "trial_id":    trial_id,
        "status":      status,
        "reward":      float(reward) if reward is not None else -1.0,
        "error_trace": error_trace,
        "metrics": {
            "ref_time_ms":   ref_time,
            "cuda_time_ms":  cuda_time,
            # Defaults — overwritten by extra_fields on SUCCESS / WRONG_MATH
            "allclose":      False,
            "max_abs_diff":  float("inf"),
        },
    }
    if extra_fields:
        # Merge extra_fields into metrics where keys overlap, else top-level
        for k, v in extra_fields.items():
            if k in payload["metrics"]:
                payload["metrics"][k] = v
            else:
                payload[k] = v
    return payload

# ========== Source pre-processing ==========
def fix_mismatched_kernel_name(code: str):
    defined = re.findall(r'__global__\s+\w+\s+(\w+)\s*\(', code)
    called  = re.findall(r'(\w+)\s*<<<', code)
    if len(defined) == 1 and len(called) == 1 and defined[0] != called[0]:
        old, new = called[0], defined[0]
        fixed = re.sub(rf'\b{re.escape(old)}\s*<<<', f'{new}<<<', code)
        return fixed, old, new
    return code, None, None

def get_undefined_kernels(code: str) -> set:
    defined = set(re.findall(r'__global__\s+\w+\s+(\w+)\s*\(', code))
    called  = set(re.findall(r'(\w+)\s*<<<', code))
    return called - defined

def short_hash(s: str, length: int = 8) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:length]

# ========== Compilation ==========
def compile_extension(cuda_code: str, ext_name: str):
    """
    ext_name must be a valid C identifier and unique per compilation unit.
    Caller is responsible for uniqueness.
    """
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
    import subprocess
    nvcc = shutil.which("nvcc") or "/opt/conda/bin/nvcc"
    torch_dir = None
    try:
        import torch
        torch_dir = os.path.dirname(torch.__file__)
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
            # compile only — do NOT link, avoids all the missing -ltorch etc.
            "-c", cu_path, "-o", os.path.join(build_dir, "probe.o"),
        ]
        if torch_dir:
            cmd += [
                "-isystem", os.path.join(torch_dir, "include"),
                "-isystem", os.path.join(torch_dir, "include", "torch", "csrc", "api", "include"),
            ]
        # Add python include
        import sysconfig
        py_inc = sysconfig.get_path("include")
        if py_inc:
            cmd += ["-isystem", py_inc]

        probe = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if probe.returncode == 0:
            return ""   # compiled cleanly — warnings are not errors
        return probe.stderr or probe.stdout or "(no output)"
    except Exception as e:
        return f"(probe failed: {e})"
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)

        
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

# ========== Op-aware input generation ==========
# Keywords → (generator_fn)  All tensors are created on CUDA.
# Each generator returns a tuple that can be unpacked directly into the kernel.
#
# Strategy: infer from op_name which shape family to use.
# Falls back to a generic 1-D float tensor if nothing matches.

def _make_inputs_for_op(op_name: str):
    op = op_name.lower()

    # ── Must come first — specific ops that share keywords with generic ones ─
    if "masked_cumsum" in op:
        x    = torch.randn(256, 1024, device="cuda")
        mask = torch.ones(256, 1024, dtype=torch.bool, device="cuda")
        return [x, mask, 1]

    if "cumsum" in op or "cumprod" in op or "scan" in op:
        return (torch.randn(256, 1024, device="cuda"),)

    if "diagonal" in op and any(k in op for k in ["matmul", "gemm", "mat_mul"]):
        return (torch.randn(512, device="cuda"),       # 1D diagonal
                torch.randn(512, 512, device="cuda"))  # 2D matrix

    # ── Matrix multiply family ──────────────────────────────────────────────
    if any(k in op for k in ["matmul", "gemm", "matrix_mul",
                              "mat_mul", "linear"]):
        return (torch.randn(512, 512, device="cuda"),
                torch.randn(512, 512, device="cuda"))

    # ── Batch matmul ────────────────────────────────────────────────────────
    if "bmm" in op or ("batch" in op and "matmul" in op):
        return (torch.randn(8, 256, 256, device="cuda"),
                torch.randn(8, 256, 256, device="cuda"))

    # ── Softmax / log-softmax ───────────────────────────────────────────────
    if "softmax" in op:
        return (torch.randn(128, 1024, device="cuda"),)

    # ── RMSNorm — must be before layernorm block ────────────────────────────
    if "rmsnorm" in op or "rms_norm" in op:
        return [torch.randn(4, 256, 768, device="cuda"), 1e-6]

    # ── LayerNorm ───────────────────────────────────────────────────────────
    if any(k in op for k in ["layernorm", "layer_norm"]):
        B, T, C = 4, 256, 768
        return (torch.randn(B, T, C, device="cuda"),
                torch.ones(C, device="cuda"),
                torch.zeros(C, device="cuda"))

    # ── GroupNorm / InstanceNorm ────────────────────────────────────────────
    if any(k in op for k in ["groupnorm", "group_norm", "instancenorm"]):
        return (torch.randn(4, 32, 64, 64, device="cuda"),)

    # ── Cross-entropy / NLL loss ────────────────────────────────────────────
    if any(k in op for k in ["cross_entropy", "nll", "loss"]):
        return (torch.randn(256, 1024, device="cuda"),
                torch.randint(0, 1024, (256,), device="cuda"))

    # ── Activation functions ────────────────────────────────────────────────
    if any(k in op for k in ["gelu", "silu", "relu", "sigmoid", "tanh",
                              "activation"]):
        return (torch.randn(512, 1024, device="cuda"),)

    # ── Dropout ─────────────────────────────────────────────────────────────
    if "dropout" in op:
        return (torch.randn(512, 1024, device="cuda"),)

    # ── Element-wise / add / mul / div ──────────────────────────────────────
    if any(k in op for k in ["add", "mul", "div", "elementwise",
                              "element_wise", "hadamard"]):
        return (torch.randn(512, 512, device="cuda"),
                torch.randn(512, 512, device="cuda"))

    # ── Convolution ─────────────────────────────────────────────────────────
    if any(k in op for k in ["conv", "convolution"]):
        return (torch.randn(4, 64, 32, 32, device="cuda"),
                torch.randn(64, 64, 3, 3, device="cuda"))

    # ── Attention ───────────────────────────────────────────────────────────
    if any(k in op for k in ["attention", "attn", "scaled_dot"]):
        B, H, T, D = 2, 8, 128, 64
        return (torch.randn(B, H, T, D, device="cuda"),
                torch.randn(B, H, T, D, device="cuda"),
                torch.randn(B, H, T, D, device="cuda"))

    # ── RoPE / positional embedding ─────────────────────────────────────────
    if any(k in op for k in ["rope", "rotary", "positional"]):
        return (torch.randn(4, 128, 8, 64, device="cuda"),)

    # ── Embedding / gather / scatter ────────────────────────────────────────
    if any(k in op for k in ["embedding", "gather", "scatter"]):
        return (torch.randn(10000, 256, device="cuda"),
                torch.randint(0, 10000, (32, 64), device="cuda"))

    # ── Sort / topk ─────────────────────────────────────────────────────────
    if any(k in op for k in ["sort", "topk", "top_k"]):
        return (torch.randn(256, 1024, device="cuda"),)

    # ── Reduction ───────────────────────────────────────────────────────────
    if any(k in op for k in ["reduce", "reduction", "sum", "mean", "max"]):
        return (torch.randn(256, 1024, device="cuda"),)

    # ── Transpose / permute ─────────────────────────────────────────────────
    if any(k in op for k in ["transpose", "permute"]):
        return (torch.randn(256, 512, device="cuda"),)

    # ── Default fallback ────────────────────────────────────────────────────
    log_msg("WARNING",
            f"No input spec matched op='{op_name}' — using generic 2-D fallback")
    return (torch.randn(256, 512, device="cuda"),)


def _try_inputs(cand_fn, ref_fn, inputs):
    """Run both fns with inputs. Return (ref_out, cand_out) or raise."""
    with torch.no_grad():
        ref_out  = ref_fn(*inputs)
        cand_out = cand_fn(*inputs)
    if isinstance(ref_out,  (list, tuple)): ref_out  = ref_out[0]
    if isinstance(cand_out, (list, tuple)): cand_out = cand_out[0]
    return ref_out, cand_out

def time_ms(fn, inputs, warmup=TIMING_WARMUP, runs=TIMING_RUNS) -> float:
    for _ in range(warmup):
        fn(*inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    total = 0.0
    for _ in range(runs):
        start.record()
        fn(*inputs)
        end.record()
        torch.cuda.synchronize()
        total += start.elapsed_time(end)
    return total / runs

# ========== Core evaluation ==========
def evaluate_cuda_mode(trial_id, candidate_code, reference_code, op_name="unknown"):
    # ── Pre-process ─────────────────────────────────────────────────────────
    cand_code, old_c, new_c = fix_mismatched_kernel_name(candidate_code)
    ref_code,  old_r, new_r = fix_mismatched_kernel_name(reference_code)
    if old_c: log_msg("INFO", f"Fixed candidate kernel name: {old_c} -> {new_c}")
    if old_r: log_msg("INFO", f"Fixed reference  kernel name: {old_r} -> {new_r}")
    undefined = get_undefined_kernels(cand_code) | get_undefined_kernels(ref_code)
    if undefined:
        return build_result_payload(trial_id, "SKIP_MISSING_KERNEL", -1.0,
                                    f"Undefined kernels: {undefined}", None, None)
    # ── Safe unique ext_names (always short, always unique per trial side) ──
    h        = short_hash(trial_id)
    cand_ext = f"k_{h}_c"
    ref_ext  = f"k_{h}_r"
    # ── Compile ─────────────────────────────────────────────────────────────
    try:
        cand_mod = compile_extension(cand_code, cand_ext)
        ref_mod  = compile_extension(ref_code,  ref_ext)
        cand_fn  = find_kernel_callable(cand_mod)
        ref_fn   = find_kernel_callable(ref_mod)
    except Exception as e:
        real_err = get_real_nvcc_error(candidate_code)
        return build_result_payload(trial_id, "COMPILATION_ERROR", -1.0,
                                    f"cpp_load: {e}\nnvcc:\n{real_err}", None, None)
    # ── Op-aware input generation ────────────────────────────────────────────
    inputs = _make_inputs_for_op(op_name)
    log_msg("INFO", f"op='{op_name}' → inputs: {[tuple(t.shape) if hasattr(t, 'shape') else t for t in inputs]}")
    # ── Correctness ──────────────────────────────────────────────────────────
    try:
        ref_out, cand_out = _try_inputs(cand_fn, ref_fn, inputs)
    except Exception as e:
        # Attempt to recover GPU context before returning
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
        return build_result_payload(trial_id, "RUNTIME_ERROR", -1.0,
                                    f"Kernel exec: {e}", None, None)
        
    max_diff = (cand_out.float() - ref_out.float()).abs().max().item()
    allclose  = torch.allclose(cand_out.float(), ref_out.float(), atol=ATOL, rtol=RTOL)
    if not allclose:
        return build_result_payload(
            trial_id, "WRONG_MATH", 0.0,
            f"max_diff={max_diff:.6f}", None, None,
            extra_fields={"allclose": False, "max_abs_diff": max_diff},
        )
    # ── Timing ───────────────────────────────────────────────────────────────
    try:
        cand_ms = time_ms(cand_fn, inputs)
        ref_ms  = time_ms(ref_fn,  inputs)
        speedup = ref_ms / cand_ms if cand_ms > 0 else -1.0
    except Exception as e:
        return build_result_payload(trial_id, "SYSTEM_ERROR", -1.0,
                                    f"Timing: {e}", None, None)
    reward = speedup if REWARD_MODE == "speedup" else 1.0
    return build_result_payload(
        trial_id, "SUCCESS", reward, None, ref_ms, cand_ms,
        extra_fields={
            "speedup":      speedup,
            "allclose":     True,
            "max_abs_diff": max_diff,
            "op_name":      op_name,
        },
    )

    
# ========== Trial processing ==========
def process_trial(trial_id):
    cand_path   = os.path.join(PENDING_DIR,   f"{trial_id}_candidate.cu")
    ref_path    = os.path.join(PENDING_DIR,   f"{trial_id}_reference.cu")
    meta_path   = os.path.join(PENDING_DIR,   f"{trial_id}_meta.json")
    result_path = os.path.join(COMPLETED_DIR, f"{trial_id}_results.json")

    if os.path.exists(result_path):
        cleanup_artifacts([cand_path, ref_path, meta_path])
        return

    if not is_file_stable(cand_path) or not is_file_stable(ref_path):
        return

    with open(cand_path, "r") as f: cand_code = f.read()
    with open(ref_path,  "r") as f: ref_code  = f.read()

    # op_name is written into an optional sidecar _meta.json by the notebook
    op_name = "unknown"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            op_name = json.load(f).get("op_name", "unknown")

    result = evaluate_cuda_mode(trial_id, cand_code, ref_code, op_name=op_name)
    atomic_write_json(result_path, result, trial_id)
    cleanup_artifacts([cand_path, ref_path, meta_path])
    log_msg("INFO",
            f"Completed {trial_id} [{op_name}] → {result['status']}  "
            f"reward={result['reward']:.4f}")

# ========== Summary ==========
def print_summary():
    results = []
    for fn in os.listdir(COMPLETED_DIR):
        if fn.endswith("_results.json") and not fn.startswith("temp_"):
            with open(os.path.join(COMPLETED_DIR, fn)) as f:
                results.append(json.load(f))
    if not results:
        log_msg("INFO", "No results to summarise.")
        return

    from collections import Counter
    statuses = Counter(r["status"] for r in results)

    print("\n" + "=" * 100)
    print(f"REWARD_MODE = {REWARD_MODE}")
    print(f"{'Trial ID':<30} {'Op':<20} {'Status':<22} {'Speedup':>8} {'Reward':>8}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: -(x.get("reward", -1))):
        sid = r["trial_id"][:28]
        op  = (r.get("op_name") or "?")[:18]
        spd = r.get("speedup")
        perf = f"{spd:.3f}x" if spd is not None else "—"
        print(f"{sid:<30} {op:<20} {r['status']:<22} {perf:>8} {r['reward']:>8.3f}")

    print("=" * 100)
    print("Status breakdown:", dict(statuses))
    successes = [r for r in results if r["status"] == "SUCCESS"]
    if successes:
        avg = sum(r["reward"] for r in successes) / len(successes)
        print(f"SUCCESS: {len(successes)}  avg reward: {avg:.3f}")

# ========== Main ==========
def main():
    ensure_directories()
    log_msg("INFO", f"Evaluator started (BASELINE_MODE={BASELINE_MODE}). "
                    f"Watching {PENDING_DIR}/ ...")

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
        except Exception as e:
            log_msg("CRITICAL", f"Loop error: {e}\n{traceback.format_exc()}")
            if "CUDA error" in str(e) or "misaligned" in str(e) or "AcceleratorError" in str(type(e).__name__):
                log_msg("CRITICAL", "CUDA context poisoned — marking pending trials and restarting process")
                # Mark all currently pending trials as failed so they don't block
                try:
                    for tid in list_trial_ids():
                        result_path = os.path.join(COMPLETED_DIR, f"{tid}_results.json")
                        if not os.path.exists(result_path):
                            payload = build_result_payload(
                                tid, "RUNTIME_ERROR", -1.0,
                                f"CUDA context poisoned: {e}", None, None)
                            atomic_write_json(result_path, payload, tid)
                            cleanup_artifacts([
                                os.path.join(PENDING_DIR, f"{tid}_candidate.cu"),
                                os.path.join(PENDING_DIR, f"{tid}_reference.cu"),
                                os.path.join(PENDING_DIR, f"{tid}_meta.json"),
                            ])
                            log_msg("INFO", f"Marked {tid} as RUNTIME_ERROR (CUDA context failure)")
                except Exception as cleanup_err:
                    log_msg("CRITICAL", f"Cleanup failed: {cleanup_err}")
                # Hard restart — re-exec this process with a fresh CUDA context
                log_msg("INFO", "Re-executing evaluator process...")
                os.execv(sys.executable, [sys.executable] + sys.argv)
            time.sleep(POLL_INTERVAL_SECONDS)

    print_summary()

if __name__ == "__main__":
    main()