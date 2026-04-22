#!/usr/bin/env python3
"""
Script 2: Hardware Evaluation Environment for CUDA Kernel RL Optimization (Single Harness)

This script continuously monitors `pending_kernels/` for files named:
  - [trial_id]_harness.cu

For each trial, it:
  1) Compiles the single harness file (which contains both baseline and optimized kernels).
  2) Runs the executable with a strict timeout to verify correctness (checks for MATH_FAILED).
  3) Profiles the executable with Nsight Compute (`ncu --launch-count 2`) to capture both kernels.
  4) Parses the single CSV output to extract baseline and optimized metrics.
  5) Computes reward = baseline_time_ns / optimized_time_ns.
  6) Atomically writes JSON result to completed_results/[trial_id]_results.json.
  7) Cleans up input .cu file and executable.
"""

import json
import os
import signal
import subprocess
import time
from io import StringIO

import pandas as pd


# ------------------------------
# Configuration
# ------------------------------
PENDING_DIR = "pending_kernels"
COMPLETED_DIR = "completed_results"
POLL_INTERVAL_SECONDS = 1.0
RUN_TIMEOUT_SECONDS = 5.0
PROFILE_TIMEOUT_SECONDS = 20.0
FILE_STABLE_WAIT_SECONDS = 0.3
KEEP_FAILED_ARTIFACTS = True

TIME_METRIC = "gpu__time_duration.sum"
OPTIMIZED_METRICS = [
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
]


# ------------------------------
# Utility helpers
# ------------------------------
def ensure_directories() -> None:
    os.makedirs(PENDING_DIR, exist_ok=True)
    os.makedirs(COMPLETED_DIR, exist_ok=True)


def list_trial_ids() -> list:
    """Return sorted trial IDs for files ending with `_harness.cu`."""
    trial_ids = []
    for name in os.listdir(PENDING_DIR):
        if name.endswith("_harness.cu"):
            trial_id = name[: -len("_harness.cu")]
            trial_ids.append(trial_id)
    trial_ids.sort()
    return trial_ids


def parse_ncu_csv_to_dataframe(ncu_stdout: str) -> pd.DataFrame:
    """Parse Nsight Compute CSV output by locating the header line."""
    lines = ncu_stdout.splitlines()
    start_index = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("\"ID\"") or stripped.startswith("ID"):
            start_index = i
            break

    if start_index is None:
        raise ValueError("Failed to locate CSV header in ncu output.")

    csv_text = "\n".join(lines[start_index:])
    return pd.read_csv(StringIO(csv_text), engine="python", on_bad_lines="skip")


def detect_columns(df: pd.DataFrame) -> tuple:
    """Detect metric and kernel name columns from ncu CSV dataframe."""
    name_candidates = ["Metric Name", "Name", "metric_name"]
    value_candidates = ["Metric Value", "Value", "metric_value"]
    kernel_candidates = ["Kernel Name", "Kernel", "kernel_name"]

    metric_name_col = next((c for c in name_candidates if c in df.columns), None)
    metric_value_col = next((c for c in value_candidates if c in df.columns), None)
    kernel_name_col = next((c for c in kernel_candidates if c in df.columns), None)

    if metric_name_col is None or metric_value_col is None or kernel_name_col is None:
        raise ValueError(f"Could not detect required ncu CSV columns. Columns found: {list(df.columns)}")

    return metric_name_col, metric_value_col, kernel_name_col


def extract_metric_by_kernel(df: pd.DataFrame, kernel_substring: str, metric_name: str) -> float:
    """Extract a numeric metric filtered by kernel name substring and metric name."""
    metric_name_col, metric_value_col, kernel_name_col = detect_columns(df)
    
    # Filter by both Kernel Name containing the substring AND Metric Name matching exactly
    matched = df[
        df[kernel_name_col].str.contains(kernel_substring, case=False, na=False) & 
        (df[metric_name_col] == metric_name)
    ]

    if matched.empty:
        raise ValueError(f"Metric '{metric_name}' for kernel '{kernel_substring}' not found.")

    raw_value = matched.iloc[0][metric_value_col]
    text = str(raw_value).strip().replace(",", "")
    return float(text)


def extract_metric_for_kernel_name(df: pd.DataFrame, kernel_name: str, metric_name: str) -> float:
    """Extract metric value by exact kernel name and metric name."""
    metric_name_col, metric_value_col, kernel_name_col = detect_columns(df)
    matched = df[(df[kernel_name_col].astype(str) == str(kernel_name)) & (df[metric_name_col] == metric_name)]

    if matched.empty:
        raise ValueError(f"Metric '{metric_name}' for kernel '{kernel_name}' not found.")

    raw_value = matched.iloc[0][metric_value_col]
    text = str(raw_value).strip().replace(",", "")
    return float(text)


def choose_kernel_names(df: pd.DataFrame) -> tuple:
    """
    Choose baseline and optimized kernel names robustly.

    Priority:
    1) Explicit name matching containing 'baseline' and 'optimized'
    2) Fallback to first and last time-metric kernel launches
    """
    metric_name_col, _, kernel_name_col = detect_columns(df)
    time_rows = df[df[metric_name_col] == TIME_METRIC]

    if time_rows.empty:
        raise ValueError(f"No '{TIME_METRIC}' rows found in ncu output.")

    kernel_series = time_rows[kernel_name_col].astype(str)
    baseline_candidates = kernel_series[kernel_series.str.contains("baseline", case=False, na=False)]
    optimized_candidates = kernel_series[kernel_series.str.contains("optimized", case=False, na=False)]

    if not baseline_candidates.empty and not optimized_candidates.empty:
        return baseline_candidates.iloc[0], optimized_candidates.iloc[0]

    ordered_kernels = list(dict.fromkeys(kernel_series.tolist()))
    if len(ordered_kernels) >= 2:
        return ordered_kernels[0], ordered_kernels[-1]

    raise ValueError(
        "Unable to disambiguate baseline/optimized kernels from ncu output. "
        "Ensure two launches are profiled and kernel names are distinguishable."
    )


def is_file_stable(path: str, wait_seconds: float = FILE_STABLE_WAIT_SECONDS) -> bool:
    """Return True when file size/mtime stay unchanged over a short interval."""
    if not os.path.exists(path):
        return False

    try:
        before = os.stat(path)
        time.sleep(wait_seconds)
        after = os.stat(path)
    except FileNotFoundError:
        return False

    return before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns


def acquire_trial_lock(trial_id: str) -> str | None:
    """Create an exclusive lock file for a trial; return lock path when acquired."""
    lock_path = os.path.join(PENDING_DIR, f".{trial_id}.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return lock_path
    except FileExistsError:
        return None


def release_trial_lock(lock_path: str | None) -> None:
    if not lock_path:
        return
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass


def run_command(cmd: list, timeout: float = None, enforce_process_group: bool = False) -> subprocess.CompletedProcess:
    """Run a subprocess command and optionally enforce process-group kill on timeout."""
    if enforce_process_group:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=os.setsid
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return subprocess.CompletedProcess(args=cmd, returncode=process.returncode, stdout=stdout, stderr=stderr)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output=stdout, stderr=stderr)

    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def build_result_payload(trial_id, status, reward, error_trace, baseline_time, optimized_time, occ, comp, mem) -> dict:
    """Build result JSON payload."""
    return {
        "trial_id": trial_id,
        "status": status,
        "reward": float(reward),
        "error_trace": error_trace,
        "metrics": {
            "baseline_time_ns": float(baseline_time) if baseline_time is not None else None,
            "optimized_time_ns": float(optimized_time) if optimized_time is not None else None,
            "occupancy_pct": float(occ) if occ is not None else None,
            "compute_pct": float(comp) if comp is not None else None,
            "memory_pct": float(mem) if mem is not None else None,
        },
    }


def atomic_write_json(result_path: str, payload: dict, trial_id: str) -> None:
    temp_path = os.path.join(COMPLETED_DIR, f"temp_{trial_id}.json")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.rename(temp_path, result_path)


def cleanup_artifacts(paths: list) -> None:
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


def log_msg(level: str, message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    target = os.sys.stderr if level == "CRITICAL" else os.sys.stdout
    print(f"[{ts}] [{level}] {message}", file=target, flush=True)


def compact_text(text: str, max_chars: int = 2000) -> str:
    """Compact command output for embedding in JSON error traces."""
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars] + "\n...[truncated]"


# ------------------------------
# Trial processing pipeline
# ------------------------------
def process_trial(trial_id: str) -> None:
    harness_cu = os.path.join(PENDING_DIR, f"{trial_id}_harness.cu")
    executable = os.path.join(PENDING_DIR, f"{trial_id}_exec")
    result_path = os.path.join(COMPLETED_DIR, f"{trial_id}_results.json")

    if os.path.exists(result_path):
        log_msg("INFO", f"Skipping already-completed trial: {trial_id}")
        cleanup_artifacts([harness_cu, executable])
        return

    if not os.path.exists(harness_cu):
        return

    if not is_file_stable(harness_cu):
        log_msg("INFO", f"Deferring trial {trial_id}: input file still being written.")
        return

    lock_path = acquire_trial_lock(trial_id)
    if lock_path is None:
        return

    log_msg("INFO", f"Processing trial: {trial_id}")

    metrics = {"base_t": None, "opt_t": None, "occ": None, "comp": None, "mem": None}
    status = "SUCCESS"
    reward = -1.0
    error_trace = None

    try:
        # Step 1: Compile the unified harness
        compile_res = run_command(["nvcc", harness_cu, "-o", executable])
        if compile_res.returncode != 0:
            status, reward, error_trace = "COMPILATION_ERROR", -1.0, compile_res.stderr.strip()
            return finalize_trial(
                trial_id,
                result_path,
                status,
                reward,
                error_trace,
                metrics,
                [harness_cu, executable],
                keep_artifacts=True,
            )

        # Step 2: Correctness + Timeout Check (Fast Run)
        try:
            run_result = run_command([executable], timeout=RUN_TIMEOUT_SECONDS, enforce_process_group=True)
        except subprocess.TimeoutExpired:
            status, reward, error_trace = "TIMEOUT", -0.5, f"Executable timed out after {RUN_TIMEOUT_SECONDS}s."
            return finalize_trial(trial_id, result_path, status, reward, error_trace, metrics, [harness_cu, executable])

        combined_output = f"{run_result.stdout}\n{run_result.stderr}"

        if "MATH_FAILED" in combined_output:
            status, reward, error_trace = "WRONG_MATH", 0.0, "Executable reported MATH_FAILED."
            return finalize_trial(trial_id, result_path, status, reward, error_trace, metrics, [harness_cu, executable])
        elif run_result.returncode != 0:
            status, reward, error_trace = "RUNTIME_ERROR", -1.0, (
                f"Runtime error. Code: {run_result.returncode}. stderr:\n{run_result.stderr.strip()}"
            )
            return finalize_trial(
                trial_id,
                result_path,
                status,
                reward,
                error_trace,
                metrics,
                [harness_cu, executable],
                keep_artifacts=True,
            )

        # Step 3: Nsight Compute Dual Profiling
        ncu_cmd = ["ncu", "--launch-count", "2", "--csv", "--metrics", ",".join(OPTIMIZED_METRICS), executable]
        ncu_res = run_command(ncu_cmd, timeout=PROFILE_TIMEOUT_SECONDS, enforce_process_group=True)
        
        if ncu_res.returncode != 0:
            stderr_text = compact_text(ncu_res.stderr)
            stdout_text = compact_text(ncu_res.stdout)
            status, reward, error_trace = (
                "SYSTEM_ERROR",
                -1.0,
                f"ncu failed. stderr:\n{stderr_text}\nstdout:\n{stdout_text}",
            )
            return finalize_trial(
                trial_id,
                result_path,
                status,
                reward,
                error_trace,
                metrics,
                [harness_cu, executable],
                keep_artifacts=True,
            )

        # Step 4: Parse CSV and Extract metrics safely
        df = parse_ncu_csv_to_dataframe(ncu_res.stdout)

        baseline_kernel, optimized_kernel = choose_kernel_names(df)
        metrics["base_t"] = extract_metric_for_kernel_name(df, baseline_kernel, TIME_METRIC)
        metrics["opt_t"] = extract_metric_for_kernel_name(df, optimized_kernel, TIME_METRIC)
        metrics["occ"] = extract_metric_for_kernel_name(
            df, optimized_kernel, "sm__warps_active.avg.pct_of_peak_sustained_active"
        )
        metrics["comp"] = extract_metric_for_kernel_name(
            df, optimized_kernel, "sm__throughput.avg.pct_of_peak_sustained_elapsed"
        )
        metrics["mem"] = extract_metric_for_kernel_name(
            df, optimized_kernel, "dram__throughput.avg.pct_of_peak_sustained_elapsed"
        )

        # Step 5: Reward Calculation
        if metrics["base_t"] <= 0.0 or metrics["opt_t"] <= 0.0:
            status, reward, error_trace = "SYSTEM_ERROR", -1.0, "Invalid baseline or optimized time (<= 0) from ncu."
        else:
            reward = metrics["base_t"] / metrics["opt_t"]
            status, error_trace = "SUCCESS", None

        return finalize_trial(trial_id, result_path, status, reward, error_trace, metrics, [harness_cu, executable])

    except subprocess.TimeoutExpired:
        status, reward, error_trace = "TIMEOUT", -0.5, f"ncu timed out after {PROFILE_TIMEOUT_SECONDS}s."
        return finalize_trial(
            trial_id,
            result_path,
            status,
            reward,
            error_trace,
            metrics,
            [harness_cu, executable],
            keep_artifacts=True,
        )
    except Exception as exc:
        status, reward, error_trace = "SYSTEM_ERROR", -1.0, f"Unhandled exception: {exc}"
        log_msg("CRITICAL", f"Trial {trial_id}: {error_trace}")
        finalize_trial(
            trial_id,
            result_path,
            status,
            reward,
            error_trace,
            metrics,
            [harness_cu, executable],
            keep_artifacts=True,
        )
    finally:
        release_trial_lock(lock_path)


def finalize_trial(trial_id, result_path, status, reward, error_trace, m, cleanup_paths, keep_artifacts=False) -> None:
    payload = build_result_payload(trial_id, status, reward, error_trace, m["base_t"], m["opt_t"], m["occ"], m["comp"], m["mem"])
    try:
        atomic_write_json(result_path, payload, trial_id)
        log_msg("INFO", f"Finished {trial_id} | Status: {status} | Reward: {reward:.4f}")
    finally:
        if not (keep_artifacts and KEEP_FAILED_ARTIFACTS):
            cleanup_artifacts(cleanup_paths)


# ------------------------------
# Main watcher loop
# ------------------------------
def main() -> None:
    ensure_directories()
    log_msg("INFO", "Script 2 evaluator started. Monitoring pending_kernels/ for _harness.cu files...")

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