#!/usr/bin/env python3
"""Benchmark eager PyTorch forward inference using CUDA synchronization."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from gpu_lenet.checkpoint import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/lenet5.pt")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", default="results/pytorch_inference.json")
    args = parser.parse_args()
    if args.batch_size < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("batch/iterations must be positive; warmup >= 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device(args.device)
    model = load_model(args.checkpoint).to(device)
    inputs = torch.randn(args.batch_size, 1, 28, 28, device=device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times_ms = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(inputs)
        synchronize()
        for _ in range(args.iterations):
            synchronize()
            start = time.perf_counter()
            model(inputs)
            synchronize()
            times_ms.append(1000 * (time.perf_counter() - start))
    report = {
        "backend": "PyTorch eager inference", "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__, "batch_size": args.batch_size,
        "warmup": args.warmup, "iterations": args.iterations,
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms), "max_ms": max(times_ms),
        "method": "synchronized end-to-end PyTorch forward call",
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
