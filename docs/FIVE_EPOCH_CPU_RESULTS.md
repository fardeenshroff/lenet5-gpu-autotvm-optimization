# Five-Epoch MNIST Training — Verified GitHub Actions Run

**Date:** November 22, 2026  
**Status:** Successful online execution  
**Execution:** Linux CPU runner, Intel Xeon Platinum 8573C, PyTorch 2.15.0  
**Evidence:** [GitHub Actions run](https://github.com/fardeenshroff/lenet5-gpu-autotvm-optimization/actions/runs/45048386353) · [Raw JSON, checkpoint artifact](https://github.com/fardeenshroff/lenet5-gpu-autotvm-optimization/actions/runs/45048386353/artifacts/14764786383)

## Measured accuracy progression

| Epoch | MNIST test accuracy | Mean training loss | Wall time (s) |
|---:|---:|---:|---:|
| 1 | 94.92% | 0.5435 | 8.136 |
| 2 | 97.39% | 0.1790 | 8.008 |
| 3 | 98.06% | 0.1171 | 7.967 |
| 4 | 98.30% | 0.0942 | 7.903 |
| 5 | 98.58% | 0.0797 | 7.862 |

**Final test accuracy: 98.58%.** Overall five-epoch wall time: **39.878 seconds** (includes test evaluation).  
**PyTorch CPU inference:** median **0.1795 ms**, mean **0.1851 ms**; batch size 1, 30 warmups, 200 measured passes.

## Method and limitations

- This is a **new, independently executed CPU benchmark** on a GitHub-hosted runner, not a new NVIDIA GPU result.
- CPU and NVIDIA GPU are benchmarked separately. The old one-epoch Colab run reached **94.80% GPU test accuracy**, but its hardware and execution context differ, so the CPU five-epoch result is not a controlled cross-device accuracy comparison.
- The earlier one-epoch GitHub CPU run used batch size 512, while this five-epoch run uses 256. Accordingly, differences cannot be attributed solely to training duration.
- To obtain **five-epoch NVIDIA Tesla T4 measurements**, rerun the [updated Colab notebook](https://colab.research.google.com/github/fardeenshroff/lenet5-gpu-autotvm-optimization/blob/main/notebooks/gpu_benchmark_colab.ipynb) with a GPU runtime.

## Machine-readable evidence

[Verified JSON summary](../results/verified_cloud_cpu_5epochs_2026-11-22.json) · [GitHub Actions source](../.github/workflows/benchmark-cpu-five-epochs.yml)
