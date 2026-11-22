# Verified Online Benchmark — November 22, 2026

**Result:** <https://github.com/fardeenshroff/lenet5-gpu-autotvm-optimization/actions/runs/45046364801> — GitHub Actions succeeded.

## New results (cloud CPU)

| Test | Recorded result |
|---|---:|
| MNIST LeNet-5, 1 training epoch | **92.34% test accuracy** |
| Total training loop time (1 epoch, includes evaluation) | **7.6841 s** |
| PyTorch eager CPU inference (mean, batch 1) | **0.1482 ms** |
| PyTorch eager CPU inference (median, batch 1) | **0.1443 ms** |
| Minimum / maximum CPU latency | 0.1382 / 0.1931 ms |
| Model unit tests | 3 passed |

**Runner:** Linux, Intel Xeon 6975P-C, 4 logical CPUs, Python 3.11.18, PyTorch 2.15.0+cu132, **CUDA unavailable**. The benchmark used 30 warmups and 200 timed inferences on a trained LeNet-5 checkpoint. The inference input is synthetic, so this is a performance measurement, not a fresh accuracy measurement. Accuracy was measured on the MNIST test set during training.

**Traceability:** [Workflow run](https://github.com/fardeenshroff/lenet5-gpu-autotvm-optimization/actions/runs/45046364801) · [Raw JSON + checkpoint artifact](https://github.com/fardeenshroff/lenet5-gpu-autotvm-optimization/actions/runs/45046364801/artifacts/14762724896) · [Machine-readable benchmark](../results/verified_cloud_cpu_2026-11-22.json)

## Separate NVIDIA GPU evaluation

The free GitHub Actions runner used for the measurements **has no NVIDIA GPU**, so no CUDA, TVM CUDA, or AutoTVM performance number is claimed for this run. A [ready-to-run Google Colab notebook](https://colab.research.google.com/github/fardeenshroff/lenet5-gpu-autotvm-optimization/blob/main/notebooks/gpu_benchmark_colab.ipynb) enables new GPU measurements on a GPU-enabled runtime.

## Historical research

The original research report documents a **2.39x** GPU-versus-PyTorch training speedup under its own experimental conditions. The new cloud CPU measurements are **not a replication of that training-speedup experiment**. See [historical study findings](RESEARCH_RESULTS.md).
