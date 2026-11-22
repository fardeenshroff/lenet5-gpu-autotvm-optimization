"""Benchmark report helpers; pure Python so they can be verified in CI."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


def summarize_ms(samples: list[float]) -> dict[str, float | int]:
    """Each sample is synchronized elapsed milliseconds per forward pass."""
    if not samples or any(x <= 0 for x in samples):
        raise ValueError("Nonempty positive samples are required")
    return {
        "samples": len(samples),
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def calculate_speedup(reference_ms: float, optimized_ms: float) -> float:
    if reference_ms <= 0 or optimized_ms <= 0:
        raise ValueError("Latencies must be positive")
    return reference_ms / optimized_ms


def save_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
