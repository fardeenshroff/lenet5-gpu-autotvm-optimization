# LeNet-5 GPU Optimization with CUDA, TVM & AutoTVM

**Researcher:** [Fardeen Shroff](https://github.com/fardeenshroff)  
**Focus:** GPU acceleration · hardware-aware compilation · CUDA/C++ · inference profiling

This repository contains runnable PyTorch, CUDA/C++, and legacy TVM/AutoTVM
experiments, automated CPU tests, and benchmark workflows. It documents
my LeNet-5 acceleration research on an NVIDIA GPU and provides tools for
collecting new performance measurements.

## Validated NVIDIA Tesla T4 performance — five epochs

The Google Colab experiment on **22 November 2026** produced verified PyTorch CPU-versus-GPU measurements on MNIST. The complete results and methodology appear in the [research manuscript](paper/Batch_Size_Dependent_LeNet5_GPU_Study.md); [all nine raw JSON reports](results/colab_t4_five_epochs_2026-11-22/) are publicly available.

| Measure | Colab host CPU | NVIDIA Tesla T4 | Result |
|---|---:|---:|---:|
| Five-epoch training + evaluation | 122.274 s | 82.904 s | **1.47× GPU speedup** |
| Epoch-five MNIST test accuracy | 98.58% | 98.52% | Similar accuracy |
| Median inference, batch size 1 | 0.4494 ms | 0.3486 ms | **1.29× GPU speedup** |
| Median inference, batch size 128 | 7.4287 ms | 0.4119 ms | **18.03× GPU speedup** |

**Scope:** The 18.03× value is a synchronized **PyTorch forward-pass** comparison using the same checkpoint and batch size on both devices. This is not a newly reproduced TVM/AutoTVM compiler result, and the historical 2.39× claim is discussed separately.

## New project — LeNet-5 with Relax and MetaSchedule

The modern compiler-optimization follow-up is now available as a [standalone project](projects/lenet5-relax-metaschedule/README.md) with [full source](projects/lenet5-relax-metaschedule/src/relax_meta/benchmark.py), [free GPU Colab notebook](https://colab.research.google.com/github/fardeenshroff/lenet5-gpu-autotvm-optimization/blob/main/projects/lenet5-relax-metaschedule/notebooks/relax_metaschedule_colab.ipynb), and CPU-validated tests. It compares **PyTorch CUDA**, **Relax default GPU schedules**, and **MetaSchedule-tuned Relax** on the **same checkpoint and GPU**. The key result is **tuned Relax / untuned Relax** latency improvement, not the previous PyTorch CPU/GPU speedup.

**Execution status:** All 16 source-side tests pass; the project has no new measured MetaSchedule GPU speedup yet. Results are generated only when a real CUDA run passes correctness validation.

## Run the code online

- **CPU (automated):** [GitHub Actions benchmark](../../actions/workflows/benchmark-cpu.yml)
  trains on MNIST, records test accuracy, and measures inference latency.
- **NVIDIA GPU — five-epoch accuracy run:** [Open the updated Colab notebook](https://colab.research.google.com/github/fardeenshroff/lenet5-gpu-autotvm-optimization/blob/main/notebooks/gpu_benchmark_colab.ipynb)
  with a GPU runtime to train MNIST for **five epochs**, collect per-epoch test accuracy, and compare CUDA versus CPU training and inference. The notebook exports a separate five-epoch results ZIP.
- **Five-epoch CPU cross-check:** [GitHub Actions workflow](../../actions/workflows/benchmark-cpu-five-epochs.yml) uses free online CPU resources; it does **not** establish GPU results.
- **NVIDIA proof:** [CUDA runtime verification script](scripts/verify_cuda_runtime.py)
  executes and validates a real GPU operation, not just a GPU-presence check.
- **Laptop fallback:** [Windows/Linux instructions](docs/RUN_ON_LAPTOP.md) for local NVIDIA hardware.
- **Raw scripts:** See [setup](docs/SETUP.md) to use your own Linux/NVIDIA environment.

The historical study's measurements and newly generated benchmarks are
reported separately to make results independently interpretable.

## Verified five-epoch online MNIST benchmark — 98.58% accuracy

A completed [GitHub Actions five-epoch run](https://github.com/fardeenshroff/lenet5-gpu-autotvm-optimization/actions/runs/45048386353) reached **98.58% MNIST test accuracy** on a cloud **CPU** (Intel Xeon Platinum 8573C) using PyTorch and batch size 256. The five-epoch wall time was **39.88 s**. The accuracy progressed from **94.92% → 97.39% → 98.06% → 98.30% → 98.58%**.

[Five-epoch results report](docs/FIVE_EPOCH_CPU_RESULTS.md) · [Verified raw JSON](results/verified_cloud_cpu_5epochs_2026-11-22.json) · [Run artifact and model checkpoint](https://github.com/fardeenshroff/lenet5-gpu-autotvm-optimization/actions/runs/45048386353/artifacts/14764786383)

**Scope:** This independent GitHub Actions run measured CPU-only accuracy on another machine; the matched Colab CPU-versus-GPU comparison and its raw results appear in the NVIDIA T4 section above.

## Verified new cloud benchmark — November 22, 2026

A completed [GitHub Actions benchmark](https://github.com/fardeenshroff/lenet5-gpu-autotvm-optimization/actions/runs/45046364801) recorded the following **new CPU measurements** on an Intel Xeon 6975P-C Linux runner:

| Measurement | Verified new result |
|---|---:|
| MNIST test accuracy after 1 epoch | **92.34%** |
| CPU training loop time, 1 epoch | **7.684 s** |
| PyTorch CPU inference median, batch 1 | **0.1443 ms** |
| PyTorch CPU inference mean, batch 1 | **0.1482 ms** |
| Unit tests | **3 passed** |

[Detailed methodology and raw results](docs/ONLINE_BENCHMARK_RESULTS.md) · [Raw JSON](results/verified_cloud_cpu_2026-11-22.json)

**Scope:** The runner had no NVIDIA GPU. These new numbers are CPU-only and are not a new verification of the historical GPU training speedup.

## Original research study: reported outcomes

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

Current benchmark JSON files are generated by the scripts and published
from individual runs. The **2.39x** number comes from the original study's
**training** comparison, whereas the new CPU/GPU benchmark scripts measure
**inference** latency. Do not compare these different metrics as equivalent.

## Relevance to GPU programming

This study demonstrates cross-platform optimization reasoning, profiling,
operator scheduling, memory/layout considerations, and the CUDA execution
stack. It provides a starting point for extending kernels with shared-memory
tiling, occupancy exploration, and Nsight Compute profiling.

## License

MIT — see [LICENSE](LICENSE).
