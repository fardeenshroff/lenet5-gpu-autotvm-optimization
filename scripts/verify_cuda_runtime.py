#!/usr/bin/env python3
"""Verify actual NVIDIA CUDA execution and record a measured GPU operation.

This check must fail on CPU-only systems. It never invents GPU availability.
The CUDA-event timing is a microbenchmark for matrix multiplication, not
LeNet-5 inference and not the original study's 2.39x speedup.
"""
import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def query_nvidia_smi():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "message": "nvidia-smi not on PATH"}
    proc = subprocess.run(
        [executable, "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"], capture_output=True, text=True,
        timeout=15, check=False,
    )
    return {
        "available": proc.returncode == 0,
        "returncode": proc.returncode,
        "output": proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip(),
    }


def verify():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "CUDA not available: no accessible NVIDIA GPU in this Python runtime. "
            "On Colab: Runtime > Change runtime type > T4 GPU, then reconnect. "
            "On a local PC: check nvidia-smi and install a CUDA-enabled PyTorch build."
        )

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    device = torch.device(f"cuda:{index}")

    # Compare a genuine on-device matrix multiplication against CPU reference.
    a_cpu = torch.randn(256, 256, dtype=torch.float32)
    b_cpu = torch.randn(256, 256, dtype=torch.float32)
    cpu_result = a_cpu @ b_cpu
    a_gpu, b_gpu = a_cpu.to(device), b_cpu.to(device)
    with torch.inference_mode():
        for _ in range(10):
            _ = a_gpu @ b_gpu
        torch.cuda.synchronize(device)
        intervals = []
        gpu_result = None
        for _ in range(60):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            gpu_result = a_gpu @ b_gpu
            end.record()
            end.synchronize()
            intervals.append(float(start.elapsed_time(end)))
    assert gpu_result is not None
    gpu_result_cpu = gpu_result.cpu()
    torch.testing.assert_close(gpu_result_cpu, cpu_result, rtol=1e-3, atol=1e-3)
    error = float((gpu_result_cpu - cpu_result).abs().max().item())

    return {
        "status": "verified_cuda_execution",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device_name": torch.cuda.get_device_name(index),
        "cuda_device_index": index,
        "device_count": torch.cuda.device_count(),
        "compute_capability": [props.major, props.minor],
        "gpu_memory_total_bytes": int(props.total_memory),
        "torch_version": torch.__version__,
        "pytorch_cuda_runtime": torch.version.cuda,
        "driver_info": query_nvidia_smi(),
        "nvcc_available": shutil.which("nvcc") is not None,
        "os": platform.platform(),
        "python": platform.python_version(),
        "test_operation": "float32 256x256 GPU matrix multiplication vs CPU",
        "test_passed": True,
        "max_absolute_error": error,
        "cuda_event_kernel_latency_median_ms": statistics.median(intervals),
        "cuda_event_kernel_latency_mean_ms": statistics.mean(intervals),
        "timed_operations": len(intervals),
        "timing_scope": "CUDA events for matrix multiplication, excluding host Python overhead",
        "historical_2_39x_result_reproduced": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/gpu_runtime_verification.json")
    args = parser.parse_args()
    try:
        report = verify()
    except Exception as exc:
        print(f"GPU verification FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"GPU verification PASSED; results written to {path}")


if __name__ == "__main__":
    main()
