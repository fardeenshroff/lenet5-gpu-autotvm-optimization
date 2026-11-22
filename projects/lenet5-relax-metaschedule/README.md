# LeNet-5 with Apache TVM Relax + MetaSchedule

**A modern, reproducible compiler-optimization experiment on NVIDIA GPUs.**  
Author: Fardeen Shroff · Research project · November 2026

[![TVM](https://img.shields.io/badge/Apache%20TVM-Relax%20%2B%20MetaSchedule-327c88)](https://tvm.apache.org/docs/) [![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.13-306998)](https://www.python.org/) [![CUDA](https://img.shields.io/badge/Target-NVIDIA%20CUDA-438c64)](https://developer.nvidia.com/cuda-toolkit)

## Research question

> Holding the trained LeNet-5 checkpoint, CUDA device, tensor shape, data type, input, and timing protocol constant, **how much does MetaSchedule's measured kernel scheduling improve inference latency over a baseline compiled by TVM Relax with default GPU schedules?**

This project is **distinct from the earlier PyTorch CPU/GPU study**, which reported an 18.03× batch-128 *hardware acceleration* ratio. That result does **not** measure compiler autotuning and must not be reused as a MetaSchedule outcome.

## Implementation pipeline

```text
MNIST training (PyTorch; done once)
       │
       ▼
Trained LeNet-5 checkpoint (same weights for all variants)
       │
       ▼
PyTorch torch.export → Relax IRModule → legalization (zero pipeline)
       │
       ├── A: PyTorch CUDA eager inference (reference)
       │
       ├── B: Relax + DefaultGPUSchedule → TVM VM (untuned compiler baseline)
       │
       └── C: Relax + MetaScheduleTuneTIR + ApplyDatabase
                + DefaultGPUSchedule (fallback) → TVM VM (tuned)
                            │
                            ▼
                  Validate every output against PyTorch
                            │
                            ▼
                 Warm up → synchronize → measure
                            │
                            ▼
               JSON metrics + saved schedule database
```

The critical comparison is **C versus B** (MetaSchedule's incremental compiler contribution). The PyTorch reference A provides ecosystem context, not a substitute for that controlled test.

## Experimental matrix

| Variable | Setting |
|---|---|
| Dataset / model | MNIST, LeNet-5 variant, 28×28 grayscale, 10 classes |
| Model parameters | Single trained PyTorch checkpoint; no re-training within compilation |
| Framework | PyTorch → `torch.export` → TVM Relax / TIRx / MetaSchedule |
| Target | NVIDIA CUDA GPU selected from the active runtime |
| Batch sizes | 1 and 128, compiled and tuned **independently** |
| Dtype | FP32, static input shapes |
| MetaSchedule search | Default 128 global trials per batch, 16 per task cap |
| Timing | 20 warmups; 7 samples × 50 synchronized calls per sample |
| Output checks | `numpy.testing.assert_allclose`, atol/rtol 1e-3; fail on mismatch |
| Results | Reproducible JSON records; **no prefilled performance numbers** |

### How to run for free on NVIDIA Tesla T4

**[Open Colab notebook](https://colab.research.google.com/github/fardeenshroff/lenet5-gpu-autotvm-optimization/blob/main/projects/lenet5-relax-metaschedule/notebooks/relax_metaschedule_colab.ipynb)** · Select **Runtime → Change runtime type → T4 GPU** (or available NVIDIA GPU), then execute the cells. Free GPU availability and quotas vary.

The notebook installs a pinned modern TVM release, trains a five-epoch model or uses a compatible local checkpoint, launches the compiler experiment, and packages JSON metrics plus tuning logs for inspection. Compilation/autotuning may consume substantial GPU session time; a smaller smoke-test configuration is provided first.

### Local Linux / GPU environment

Python 3.10–3.13 and an NVIDIA GPU with a usable CUDA runtime are needed. Use an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip install 'apache-tvm[cuda,torch,meta-schedule]==0.27.0.post1'
python -m relax_meta.environment
python -m pytest -q
```

If your environment reports `tvm_cuda_available: false`, install or build TVM with CUDA support and check GPU drivers before proceeding. See [environment troubleshooting](docs/INSTALLATION.md).

Train a matching LeNet-5 checkpoint:

```bash
lenet-relax-train --epochs 5 --device cuda \
  --batch-size 256 --checkpoint checkpoints/lenet5_5epochs.pt
```

Run a smaller tuning trial first:

```bash
lenet-relax-benchmark \
  --checkpoint checkpoints/lenet5_5epochs.pt \
  --batch-sizes 1 --trials 32 --max-trials-per-task 16 \
  --output results/smoke_batch1.json
```

Then conduct the full two-shape comparison:

```bash
lenet-relax-benchmark \
  --checkpoint checkpoints/lenet5_5epochs.pt \
  --batch-sizes 1 128 --trials 128 --max-trials-per-task 16 \
  --warmup 20 --repeats 7 --number 50 \
  --output results/relax_metaschedule_comparison.json
```

**Use an existing checkpoint instead of re-training:** the checkpoint must contain `model_state_dict` matching `LeNet5` in `src/relax_meta/model.py`; this includes the earlier research repository's LeNet-5 checkpoints. Both compiled variants will receive the **same** weights.

## Reading the results

The output JSON contains **three** latency distributions for each batch size: `pytorch_cuda`, `relax_default_gpu_schedule`, and `relax_metaschedule`. If tuning is enabled, MetaSchedule must write a nonempty tuning database; otherwise the program exits without a successful results file.

| Primary metric | Exact definition | Significance |
|---|---|---|
| **MetaSchedule vs Relax baseline** | Median untuned Relax latency / median tuned Relax latency | Compiler-search benefit (may be below 1.0×) |
| Relax baseline vs PyTorch | Median PyTorch latency / median untuned Relax latency | Contextual framework comparison |
| MetaSchedule vs PyTorch | Median PyTorch latency / median tuned Relax latency | End-to-end framework comparison |
| Tuning time | MetaSchedule wall time excluding inference execution | Optimization overhead |
| Workload coverage | Extracted TIR tasks and persisted tuning records | Evidence that search occurred |

**Do not claim improvement if the measured ratio is ≤1.0×.** Negative results are useful and should be reported. Compiler target, scheduling database, hardware, batch size, and the real source checkpoint must be reported alongside any speedup.

### Measurement limitations

- The scripts measure **inference**, not training. Training uses PyTorch Adam; Relax/MetaSchedule compilation occurs **after** checkpoint generation.
- Device-to-device inputs and weights are preallocated; the timed region **excludes** host-to-device and device-to-host transfers, and excludes model compilation and tuning. Python dispatch **is included**. Comparisons are meaningful only with matching scope.
- Each timing summary contains seven samples (default); a single Colab session is **not** enough to make a statistical significance claim. Run multiple independent sessions and report uncertainty before peer-reviewed publication.
- Exact TVM APIs evolve. This project targets `apache-tvm==0.27.0.post1`; current documentation also uses the `tvm_tirx` package for target/runtime operations. Platform-dependent build fixes may be required.
- **Status:** source and CPU unit tests validated; a genuine GPU MetaSchedule run has **not yet been executed by this project**. Benchmark fields remain absent until execution.

## Project layout

```text
lenet5-relax-metaschedule/
├── src/relax_meta/
│   ├── model.py          # Same LeNet-5 architecture and checkpoint format
│   ├── train.py          # Reference MNIST training
│   ├── benchmark.py      # Relax import, compile, tune, validate and time
│   ├── environment.py    # Honest CUDA and TVM preflight
│   └── report.py         # Pure-Python speedup/report helpers
├── notebooks/
│   └── relax_metaschedule_colab.ipynb
├── tests/                # CPU-safe unit/contract tests
├── docs/
│   ├── EXPERIMENT_DESIGN.md
│   └── INSTALLATION.md
├── results/               # Output is generated by execution
├── pyproject.toml
└── README.md
```

## References and API alignment

1. [Apache TVM: End-to-End Optimize Model](https://tvm.apache.org/docs/how_to/tutorials/e2e_opt_model.html) — PyTorch `torch.export` to Relax; TIRx compile and VM.
2. [Apache TVM: MetaSchedule Search-Based Auto-Tuning](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html) — tuning tasks, passes and persistent databases.
3. [Apache TVM: Quick Start](https://tvm.apache.org/docs/get_started/tutorials/quick_start.html) — target/device APIs and VM execution.
4. [Apache TVM PyPI 0.27.0.post1](https://pypi.org/project/apache-tvm/) — current wheel and optional dependencies.
5. [Original LeNet-5 GPU study](https://github.com/fardeenshroff/lenet5-gpu-autotvm-optimization) — separate, previously verified PyTorch CPU/GPU benchmarks.

## License

MIT, as stated in [LICENSE](LICENSE). Research attribution should retain the author's name and distinguish this project from the older AutoTVM work.
