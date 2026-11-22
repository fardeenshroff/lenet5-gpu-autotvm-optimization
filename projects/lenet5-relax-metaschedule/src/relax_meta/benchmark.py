"""Reproducible LeNet-5 inference: PyTorch CUDA vs Relax vs MetaSchedule.

Only valid measurements from a real NVIDIA GPU may enter the results JSON.
This program does not tune PyTorch training or reproduce the old AutoTVM paper.

Dependencies: Apache TVM 0.27.0.post1 with Relax and MetaSchedule, PyTorch,
NVIDIA CUDA GPU. See README for exact setup and limitations.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .model import load_checkpoint
from .report import calculate_speedup, save_json, summarize_ms


def get_tvm():
    try:
        import tvm
        import tvm.tirx  # TIRx is included in the pinned Apache TVM 0.27 wheel.
        tvm_tirx = tvm  # TVM 0.27 owns target, runtime, and IR namespaces.
        from tvm import relax
        from tvm.relax.frontend.torch import from_exported_program
        from tvm.s_tir.meta_schedule.relax_integration import extract_tasks
        return tvm, tvm_tirx, relax, from_exported_program, extract_tasks
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Modern Apache TVM with Relax, TIRx and MetaSchedule is required. "
            "Install apache-tvm[cuda,torch,meta-schedule]==0.27.0.post1 "
            "with its bundled tvm.tirx module on a CUDA-capable GPU host."
        ) from exc


def validate_cuda(tvm_tirx) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA unavailable: attach an NVIDIA GPU first.")
    device = tvm_tirx.cuda(0)
    if not device.exist:
        raise RuntimeError(
            "TVM CUDA runtime unavailable although PyTorch sees the GPU. "
            "Check that the TVM wheel/build has CUDA enabled."
        )
    return device


def get_target(tvm_tirx, device, override: str | None):
    if override:
        return tvm_tirx.target.Target(override)
    return tvm_tirx.target.Target.from_device(device)


def import_relax(model, example: torch.Tensor, relax, from_exported_program):
    with torch.inference_mode():
        exported = torch.export.export(model.eval(), (example,), strict=True)
    # Pin model weights as inputs for both builds, allowing exact same weights.
    ir_mod = from_exported_program(
        exported, keep_params_as_input=True, unwrap_unit_return_tuple=True
    )
    ir_mod, params = relax.frontend.detach_params(ir_mod)
    if "main" not in params or not params["main"]:
        raise RuntimeError("Relax export did not expose the trained weights as inputs")
    return ir_mod, params["main"]


def lower_relax(ir_mod, relax, target):
    return relax.get_pipeline("zero")(ir_mod)


def default_gpu_schedule(tvm, mod, target):
    # Apply the same fallback to untuned and tuned kernels so none remain
    # unscheduled, as recommended by current Apache TVM docs.
    return tvm.s_tir.transform.DefaultGPUSchedule(target=target)(mod)


def new_virtual_machine(tvm, tvm_tirx, relax, mod, target, device):
    executable = tvm.compile(mod, target=target)
    return relax.VirtualMachine(executable, device)


def as_numpy(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError(f"Expected one model output, got {len(value)}")
        value = value[0]
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float32)
    raise TypeError(f"Unexpected Relax VM output type: {type(value)}")


def report_metadata(tvm, torch_gpu_name: str, target: object, checkpoint: str) -> dict:
    info = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "tvm_version": getattr(tvm, "__version__", "unknown"),
        "gpu_name": torch_gpu_name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "target": str(target),
        "checkpoint": checkpoint,
    }
    if shutil.which("nvidia-smi"):
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=15
        )
        if p.returncode == 0:
            info["driver_version"] = p.stdout.strip().splitlines()[0]
    return info


def count_records(directory: Path) -> int:
    records = directory / "database_tuning_record.json"
    if not records.is_file():
        return 0
    return sum(1 for line in records.read_text(encoding="utf-8").splitlines() if line.strip())


def timed_forward(fn: Callable[[], object], synchronize: Callable[[], None],
                  warmup: int, repeats: int, number: int) -> dict:
    """Wall-clock synchronized forward dispatch; no H2D/D2H transfers.

    Inputs and parameters must be allocated before this call. Each sample is
    one group's average ms/op. Synchronize after the group's final call.
    """
    for _ in range(warmup):
        fn()
    synchronize()
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        for _ in range(number):
            fn()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000 / number)
    return summarize_ms(samples)


def assert_close(actual: np.ndarray, expected: np.ndarray,
                 rtol: float, atol: float) -> dict:
    if actual.shape != expected.shape:
        raise AssertionError(f"Relax shape {actual.shape} != PyTorch {expected.shape}")
    if not np.all(np.isfinite(actual)):
        raise AssertionError("Nonfinite Relax output")
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    return {"passed": True, "rtol": rtol, "atol": atol,
            "max_absolute_error": float(np.max(np.abs(actual - expected))),
            "shape": list(actual.shape)}


def benchmark_batch(args, batch_size: int, model, device_torch,
                    tvm, tvm_tirx, relax, from_exported_program,
                    extract_tasks, tvm_device, target):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    def progress(stage: str) -> None:
        print(f"[stage][batch={batch_size}] {stage}", flush=True)

    progress("Exporting PyTorch checkpoint to Relax")
    example = torch.randn(batch_size, 1, 28, 28, dtype=torch.float32)
    ir_mod, weights = import_relax(model, example, relax, from_exported_program)
    progress("Legalizing Relax operators to TensorIR")
    lowered = lower_relax(ir_mod, relax, target)
    progress("Extracting MetaSchedule candidate tasks")
    tasks = extract_tasks(lowered, target)
    task_metadata = [{"name": str(task.task_name), "weight": int(task.weight)}
                     for task in tasks]
    # For fairness, the same Relax IR, checkpoint, target and inputs are used
    # for both compiled variants. Only the scheduling choice changes.
    progress("Applying default CUDA schedules")
    baseline_mod = default_gpu_schedule(tvm, lowered, target)
    progress("Compiling untuned Relax CUDA virtual machine")
    baseline_vm = new_virtual_machine(tvm, tvm_tirx, relax,
                                      baseline_mod, target, tvm_device)

    dir_path = Path(args.work_dir) / f"batch_{batch_size}"
    db_before = count_records(dir_path)
    tuning_sec = 0.0
    tuned_vm = None
    db_after = db_before
    if args.trials > 0:
        dir_path.mkdir(parents=True, exist_ok=True)
        start_tuning = time.perf_counter()
        progress(f"Starting MetaSchedule search ({args.trials} global trials)")
        with target:
            tuned_mod = tvm_tirx.ir.transform.Sequential([
                relax.get_pipeline("zero"),
                relax.transform.MetaScheduleTuneTIR(
                    work_dir=str(dir_path),
                    max_trials_global=args.trials,
                    max_trials_per_task=args.max_trials_per_task,
                    target=target,
                ),
                relax.transform.MetaScheduleApplyDatabase(
                    work_dir=str(dir_path), target=target
                ),
            ])(ir_mod)
        tuning_sec = time.perf_counter() - start_tuning
        progress(f"MetaSchedule search finished in {tuning_sec:.2f} s")
        db_after = count_records(dir_path)
        if db_after == 0:
            raise RuntimeError(
                f"MetaSchedule wrote zero tuning records for batch {batch_size}; "
                "refusing to label this execution as tuned."
            )
        progress("Applying default fallback to remaining untuned kernels")
        tuned_mod = default_gpu_schedule(tvm, tuned_mod, target)
        progress("Compiling MetaSchedule-tuned Relax CUDA virtual machine")
        tuned_vm = new_virtual_machine(tvm, tvm_tirx, relax,
                                       tuned_mod, target, tvm_device)

    progress("Allocating identical input and model weights on CUDA device")
    gpu_input = tvm_tirx.runtime.tensor(example.numpy(), device=tvm_device)
    gpu_weights = [tvm_tirx.runtime.tensor(
        np.asarray(p.numpy() if hasattr(p, "numpy") else p, dtype=np.float32),
        device=tvm_device) for p in weights]
    torch_model = model.to(device_torch).eval()
    x_torch = example.to(device_torch)
    with torch.inference_mode():
        torch_out = torch_model(x_torch).cpu().numpy()

    def eval_vm(vm):
        return vm["main"](gpu_input, *gpu_weights)

    progress("Validating compiled Relax outputs against PyTorch")
    baseline_out = as_numpy(eval_vm(baseline_vm))
    baseline_check = assert_close(baseline_out, torch_out, args.rtol, args.atol)
    tuned_check = None
    if tuned_vm is not None:
        tuned_out = as_numpy(eval_vm(tuned_vm))
        tuned_check = assert_close(tuned_out, torch_out, args.rtol, args.atol)
        # Also verify tuned vs untuned numerical agreement.
        assert_close(tuned_out, baseline_out, args.rtol, args.atol)

    def sync_tvm():
        tvm_device.sync()

    def sync_torch():
        torch.cuda.synchronize(device_torch)

    progress("Benchmarking warmed-up and synchronized inference")
    with torch.inference_mode():
        pytorch_timing = timed_forward(
            lambda: torch_model(x_torch), sync_torch,
            args.warmup, args.repeats, args.number
        )
    baseline_timing = timed_forward(
        lambda: eval_vm(baseline_vm), sync_tvm,
        args.warmup, args.repeats, args.number
    )
    tuned_timing = None
    if tuned_vm is not None:
        tuned_timing = timed_forward(
            lambda: eval_vm(tuned_vm), sync_tvm,
            args.warmup, args.repeats, args.number
        )

    result = {
        "batch_size": batch_size,
        "input_shape": list(example.shape),
        "tunable_tasks": task_metadata,
        "task_count": len(task_metadata),
        "tuning": {"requested_trials_global": args.trials,
                   "max_trials_per_task": args.max_trials_per_task,
                   "records_before": db_before, "records_after": db_after,
                   "new_records": db_after - db_before,
                   "elapsed_seconds": tuning_sec,
                   "directory": str(dir_path)},
        "pytorch_cuda": pytorch_timing,
        "relax_default_gpu_schedule": {"timing": baseline_timing, "correctness": baseline_check},
        "relax_metaschedule": None if tuned_vm is None else {
            "timing": tuned_timing, "correctness": tuned_check,
            "status": "measured_with_nonempty_tuning_database"
        },
        "speedups": {
            "relax_baseline_vs_pytorch": calculate_speedup(
                pytorch_timing["median_ms"], baseline_timing["median_ms"]),
        },
        "timing_scope": ("Warmups followed by grouped, synchronized GPU inference calls; "
                         "inputs and model weights pre-resident on device; "
                         "Python dispatch included; no D2H transfer or compile/tune time."),
    }
    if tuned_timing is not None:
        result["speedups"].update({
            "metaschedule_vs_relax_default": calculate_speedup(
                baseline_timing["median_ms"], tuned_timing["median_ms"]),
            "metaschedule_vs_pytorch": calculate_speedup(
                pytorch_timing["median_ms"], tuned_timing["median_ms"]),
        })
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/lenet5_5epochs.pt")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 128])
    p.add_argument("--trials", type=int, default=128,
                   help="Total MetaSchedule trial budget per batch size; 0 disables tuning")
    p.add_argument("--max-trials-per-task", type=int, default=16)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--number", type=int, default=50,
                   help="Forward passes per timed sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--target", default=None,
                   help="Override TVM target; otherwise infer CUDA device target")
    p.add_argument("--work-dir", default="tuning_logs")
    p.add_argument("--output", default="results/relax_metaschedule_comparison.json")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (not args.batch_sizes or any(b < 1 for b in args.batch_sizes)
        or any(x < 0 for x in (args.trials, args.warmup))
        or any(x < 1 for x in (args.max_trials_per_task, args.repeats, args.number))
        or args.atol < 0 or args.rtol < 0):
        parser.error("Invalid experiment arguments (positive sizes and repeats required)")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("batch-sizes must be distinct")
    model = load_checkpoint(args.checkpoint)
    tvm, tvm_tirx, relax, exporter, extract_tasks = get_tvm()
    dev = validate_cuda(tvm_tirx)
    torch_dev = torch.device("cuda:0")
    target = get_target(tvm_tirx, dev, args.target)
    output = {
        "experiment": "LeNet-5 PyTorch CUDA / TVM Relax default GPU schedule / TVM Relax MetaSchedule",
        "status": "completed_with_cuda_correctness_checks",
        "environment": report_metadata(tvm, torch.cuda.get_device_name(0), target, args.checkpoint),
        "batches": [],
        "warning": ("Tuning results measure inference compilation only. Training is done "
                    "in PyTorch; historical AutoTVM training speedups are not reproduced."),
    }
    for batch in args.batch_sizes:
        print(f"\n=== Batch size {batch}: compile, verify and benchmark ===", flush=True)
        output["batches"].append(benchmark_batch(
            args, batch, model, torch_dev, tvm, tvm_tirx, relax, exporter,
            extract_tasks, dev, target
        ))
        print(json.dumps({"batch": batch,
                          "speedups": output["batches"][-1]["speedups"]}, indent=2), flush=True)
    # Write only after every selected batch has completed, passed correctness,
    # and (if requested) produced a nonempty schedule database.
    save_json(args.output, output)
    print(f"\nVerified experiment recorded at {args.output}", flush=True)


if __name__ == "__main__":
    main()
