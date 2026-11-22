# Verify GPU Execution on Your Laptop

The NVIDIA CUDA experiments require a supported **NVIDIA GPU** with working drivers.
An Intel integrated GPU, AMD GPU, or Apple Silicon GPU does **not** run these CUDA
scripts. Do not treat a successful CPU benchmark as GPU verification.

## 1. Windows laptop (PowerShell)

From PowerShell:

```powershell
nvidia-smi
git clone https://github.com/fardeenshroff/lenet5-gpu-autotvm-optimization.git
cd lenet5-gpu-autotvm-optimization
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe scripts\verify_cuda_runtime.py
```

If PyTorch prints "CUDA not available", install a **CUDA-enabled PyTorch build**
according to the official [PyTorch installation selector](https://pytorch.org/get-started/locally/).
The correct install command depends on your driver, OS and supported toolkit.
A system CUDA toolkit is not required for PyTorch's prebuilt CUDA packages,
but `nvcc` is required for compiling the standalone `.cu` example.

When verification passes:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --device cuda --epochs 1 --output checkpoints\laptop_gpu.pt --metrics results\laptop_gpu_training.json
.\.venv\Scripts\python.exe scripts\benchmark_pytorch.py --device cuda --checkpoint checkpoints\laptop_gpu.pt --batch-size 1 --output results\laptop_cuda_inference.json
.\.venv\Scripts\python.exe scripts\benchmark_pytorch.py --device cpu --checkpoint checkpoints\laptop_gpu.pt --batch-size 1 --output results\laptop_cpu_inference.json
```

Check `results/gpu_runtime_verification.json` and other `results/*.json`
files. These are **new** measurements on your laptop, not the historical study's
2.39x GPU training speedup.

## 2. Linux with NVIDIA GPU

```bash
nvidia-smi
git clone https://github.com/fardeenshroff/lenet5-gpu-autotvm-optimization.git
cd lenet5-gpu-autotvm-optimization
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python scripts/verify_cuda_runtime.py
python scripts/train.py --device cuda --epochs 1 --output checkpoints/laptop_gpu.pt --metrics results/laptop_gpu_training.json
python scripts/benchmark_pytorch.py --device cuda --checkpoint checkpoints/laptop_gpu.pt --output results/laptop_cuda_inference.json
```

Use the official PyTorch installation selector if CUDA-enabled wheels are not
already configured.

## 3. No NVIDIA GPU on your laptop?

Use the [free Colab notebook](https://colab.research.google.com/github/fardeenshroff/lenet5-gpu-autotvm-optimization/blob/main/notebooks/gpu_benchmark_colab.ipynb).
Choose `Runtime → Change runtime type → T4 GPU → Save`, then `Runtime → Run all`.
GPU availability and quotas are controlled by Google.

## Interpreting verification correctly

A verification file reports `"status": "verified_cuda_execution"` **only**
after an actual CUDA matrix multiplication was executed and compared to a CPU
reference. `torch.cuda.is_available()` by itself is not proof the whole
LeNet-5 benchmark has executed.

The standalone CUDA/C++ microbenchmark is **separate** from the PyTorch
inference measurements and must not be combined into the same speedup ratio.
