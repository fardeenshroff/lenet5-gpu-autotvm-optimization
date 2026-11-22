# LeNet-5 GPU Optimization with CUDA, TVM & AutoTVM

**Researcher:** [Fardeen Shroff](https://github.com/fardeenshroff)  
**Focus:** GPU acceleration · hardware-aware compilation · CUDA/C++ · inference profiling

> **Project provenance:** This repository is a newly written, documented
> reproduction toolkit based on my research report, *Efficient Deep Learning
> Model Optimization on Nvidia GPUs using AutoTVM*. The original project code
> and measurement logs were not available when assembling this repository.
> Results below are from the original report, **not freshly rerun here**.

## Original study: reported outcomes

| Platform | Reported training time | Test accuracy | Relative speedup |
|---|---:|---:|---:|
| PyTorch reference | 23.27 seconds | 98.76% | 1.00x |
| TVM CPU, Intel i9-11900K | 18.78 seconds | 98.82% | 1.24x |
| TVM GPU, NVIDIA RTX 4060 | **9.75 seconds** | **98.89%** | **2.39x** |

My study explored LeNet-5 on MNIST, TVM Relay lowering and graph
transformations, CUDA target code generation, and AutoTVM kernel schedule
tuning. See [archived findings](docs/RESEARCH_RESULTS.md) for essential
methodological caveats.

## What's inside this repository

- **`src/gpu_lenet/model.py`** — reusable PyTorch LeNet-5 network.
- **`scripts/train.py`** — MNIST model training and checkpoint export.
- **`scripts/benchmark_pytorch.py`** — warmed-up, synchronized inference timing.
- **`scripts/tvm_autotvm.py`** — legacy TVM Relay CUDA compilation, optional
  AutoTVM convolution-tuning experiments, and PyTorch numerical validation.
- **`cuda/conv2d_naive.cu`** — educational CUDA/C++ convolution kernel with
  CPU reference validation and CUDA event timing. **Not** the original code.
- **`tests/`** — model shape, determinism, and backward-pass smoke tests.
- **`docs/`** — reproducibility notes and original reported research outcomes.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/train.py --epochs 2 --device auto
python scripts/benchmark_pytorch.py --device cuda
```

Use `--device cpu` for CPU-only machines. A GPU-enabled PyTorch installation
is required for CUDA benchmarking. To install **legacy** Apache TVM and
try `scripts/tvm_autotvm.py`, read [setup instructions](docs/SETUP.md).

### CUDA/C++ microbenchmark

```bash
nvcc -O3 -std=c++14 cuda/conv2d_naive.cu -o conv2d_naive
./conv2d_naive
```

The custom kernel is a **correctness-oriented educational example**, not a
performance-optimized kernel, and is separate from the historical work.

## Benchmark integrity

This repository contains **no pre-generated execution results**. Benchmark
JSON files are produced when the user runs the scripts. The historical
**2.39x training speedup is not substantiated by newly supplied inference
benchmark scripts**; replicating training-time performance requires
additional original-protocol details.

## Relevance to GPU programming

This study demonstrates cross-platform optimization reasoning, profiling,
operator scheduling, memory/layout considerations, and the CUDA execution
stack. It provides a starting point for extending kernels with shared-memory
tiling, occupancy exploration, and Nsight Compute profiling.

## License

MIT — see [LICENSE](LICENSE).
