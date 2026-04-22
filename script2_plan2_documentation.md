# Documentation: `script2_plan2.py`

## Purpose
`script2_plan2.py` is a hardware evaluation worker for CUDA kernel optimization trials.
It watches `pending_kernels/` for trial harness files (`*_harness.cu`), compiles and validates them, profiles them with Nsight Compute, computes a reward, and writes a JSON result to `completed_results/`.

---

## High-Level Workflow
For each trial (`<trial_id>_harness.cu`):

1. Wait until the file is stable (not still being written).
2. Acquire an exclusive lock file (`.<trial_id>.lock`) to avoid double-processing.
3. Compile harness with `nvcc`.
4. Run executable for correctness and timeout checks.
5. Profile with Nsight Compute (`ncu --launch-count 2 --csv ...`).
6. Parse CSV and identify baseline vs optimized kernel launches.
7. Extract timing + utilization metrics.
8. Compute reward:

   `reward = baseline_time_ns / optimized_time_ns`

9. Atomically write JSON output to `completed_results/<trial_id>_results.json`.
10. Clean up artifacts (depending on failure policy).

---

## Directory and File Contracts

### Input
- Directory: `pending_kernels/`
- Expected file format: `<trial_id>_harness.cu`

### Output
- Directory: `completed_results/`
- Result file: `<trial_id>_results.json`

### Temporary / internal
- Lock file: `pending_kernels/.<trial_id>.lock`
- Executable: `pending_kernels/<trial_id>_exec`
- Atomic temp JSON: `completed_results/temp_<trial_id>.json`

---

## Configuration Constants
- `PENDING_DIR = "pending_kernels"`
- `COMPLETED_DIR = "completed_results"`
- `POLL_INTERVAL_SECONDS = 1.0`
- `RUN_TIMEOUT_SECONDS = 5.0`
- `PROFILE_TIMEOUT_SECONDS = 20.0`
- `FILE_STABLE_WAIT_SECONDS = 0.3`
- `KEEP_FAILED_ARTIFACTS = True`

### Metrics
- Primary timing metric key: `gpu__time_duration.sum`
- Profiled metrics:
  - `gpu__time_duration.sum`
  - `sm__warps_active.avg.pct_of_peak_sustained_active`
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed`
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed`

---

## Status and Reward Behavior

### Success path
- `status = "SUCCESS"`
- `reward = baseline_time_ns / optimized_time_ns`

### Non-success statuses
- `COMPILATION_ERROR`: `nvcc` failed, reward `-1.0`
- `TIMEOUT`: executable run or profiling timed out, reward `-0.5`
- `WRONG_MATH`: executable emitted `MATH_FAILED`, reward `0.0`
- `RUNTIME_ERROR`: executable non-zero return without `MATH_FAILED`, reward `-1.0`
- `SYSTEM_ERROR`: parsing/profiling/system failures, reward `-1.0`

---

## Result JSON Schema
Each result file contains:

```json
{
  "trial_id": "...",
  "status": "SUCCESS|COMPILATION_ERROR|TIMEOUT|WRONG_MATH|RUNTIME_ERROR|SYSTEM_ERROR",
  "reward": 1.0,
  "error_trace": null,
  "metrics": {
    "baseline_time_ns": 12345.0,
    "optimized_time_ns": 6789.0,
    "occupancy_pct": 45.6,
    "compute_pct": 78.9,
    "memory_pct": 12.3
  }
}
```

Notes:
- Metric values are floats or `null` when unavailable.
- `error_trace` contains compacted stderr/stdout snippets for failures.

---

## Kernel Identification Logic
The script determines baseline vs optimized kernels using Nsight rows for `gpu__time_duration.sum`:

1. Preferred: kernel names containing `baseline` and `optimized`.
2. Fallback: first and last unique kernel launch names from time rows.

If disambiguation fails, trial is marked `SYSTEM_ERROR`.

---

## Concurrency and Robustness
- Uses lock files (`os.O_EXCL`) to prevent concurrent workers from processing the same trial.
- Verifies input file stability before processing.
- Uses process-group kill (`os.setsid` + `os.killpg`) for strict timeout cleanup.
- Writes result JSON atomically (`temp` file + `fsync` + `rename`).
- Releases lock in `finally` block.

---

## Cleanup Policy
- On normal success: removes harness `.cu` and executable.
- On selected failures with `keep_artifacts=True`: preserves artifacts if global `KEEP_FAILED_ARTIFACTS` is `True`.

This is useful for debugging failed trials.

---

## Main Loop Behavior
`main()`:
- Ensures required directories exist.
- Logs startup.
- Polls pending directory every second.
- Processes discovered trial IDs in sorted order.
- Handles `KeyboardInterrupt` gracefully.
- Logs critical loop errors and continues.

---

## Function-by-Function Reference
- `ensure_directories()`: create required input/output folders.
- `list_trial_ids()`: discover sorted trial IDs from `*_harness.cu`.
- `parse_ncu_csv_to_dataframe()`: locate CSV header in `ncu` output and parse into pandas DataFrame.
- `detect_columns()`: resolve metric/value/kernel column names across possible schema variants.
- `extract_metric_by_kernel()`: metric extraction by kernel substring (helper).
- `extract_metric_for_kernel_name()`: metric extraction by exact kernel name.
- `choose_kernel_names()`: robust baseline/optimized kernel selection.
- `is_file_stable()`: file-size + mtime stability check.
- `acquire_trial_lock()` / `release_trial_lock()`: per-trial lock management.
- `run_command()`: subprocess wrapper with optional process-group timeout handling.
- `build_result_payload()`: construct output JSON object.
- `atomic_write_json()`: safe durable write.
- `cleanup_artifacts()`: delete generated artifacts if present.
- `log_msg()`: timestamped structured logging.
- `compact_text()`: truncate long output for JSON error traces.
- `process_trial()`: complete compile/run/profile/parse/reward pipeline.
- `finalize_trial()`: persist result and apply cleanup policy.
- `main()`: continuous watcher loop.

---

## Runtime Dependencies
- Python 3.10+ (uses union type syntax `str | None`)
- `pandas`
- NVIDIA CUDA compiler: `nvcc`
- Nsight Compute CLI: `ncu`
- A GPU/runtime environment capable of executing the generated harness executable

---

## How to Run
From the `Script2/` directory:

```bash
python3 script2_plan2.py
```

Then place trial files in `pending_kernels/` using the naming pattern `<trial_id>_harness.cu`.

---

## Operational Notes
- If result JSON already exists for a trial, the trial is skipped and stale artifacts are cleaned.
- The script assumes harness execution emits `MATH_FAILED` on correctness failure.
- Profiling uses `--launch-count 2`; harness should execute baseline and optimized kernels in one run.
