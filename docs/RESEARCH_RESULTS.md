# Archived research findings — not newly benchmarked

**Author:** Fardeen Fayyaz Shroff, Illinois Institute of Technology  
**Study:** *Efficient Deep Learning Model Optimization on Nvidia GPUs using AutoTVM*

These figures are taken from the author's original written report. They are
**historical, self-reported results**, not measurements performed by this
newly authored reproduction repository.

| Implementation | Reported training time | Reported test accuracy | Reported speedup vs PyTorch |
|---|---:|---:|---:|
| PyTorch baseline | 23.27 s | 98.76% | 1.00x |
| TVM CPU (Intel i9-11900K) | 18.78 s | 98.82% | 1.24x |
| TVM GPU (NVIDIA RTX 4060) | 9.75 s | 98.89% | **2.39x** |

The source report describes MNIST LeNet-5 optimization using TVM Relay,
CUDA generation, and AutoTVM scheduling, plus layerwise investigations.

## Reproducibility boundaries

- The original paper text was available when this repo was authored, but the
  original source code and raw logs were not. All code in this repository is
  a **new implementation inspired by the documented methodology**.
- The newly provided TVM script compiles **inference only**. It is not
  evidence of the historical **training-time** speedup.
- The report's layer-level table contains formatting errors; no precise
  layerwise figures are reproduced here.
- Differences in PyTorch, CUDA, GPU drivers and TVM versions matter.

## Environment reported in the original study

- Intel Core i9-11900K; NVIDIA RTX 4060; 32 GB RAM; Ubuntu 20.04
- CUDA 11.4; cuDNN 8.1.0.77; PyTorch 1.9.0
- The source has an ambiguous TVM version entry. Confirm original toolchain
  versions before claiming a faithful historical reproduction.
