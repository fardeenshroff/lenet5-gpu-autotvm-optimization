# Installation, GPU requirements, and troubleshooting

## Verified API baseline

This project tracks the modern Apache TVM documentation for `Relax`, `MetaScheduleTuneTIR`, `MetaScheduleApplyDatabase`, `tvm.compile`, and `tvm_tirx.runtime.tensor`. The pinned PyPI wheels are `apache-tvm==0.27.0.post1` (September 2026) and `apache-tvm-tirx==0.1.0` (November 2026). The TIRx runtime is a separate distribution and must be installed explicitly. An NVIDIA GPU with a CUDA-capable TVM runtime is required to run the optimization itself.

**Two namespaces matter:**

```python
import tvm               # High-level Relax compiler API
import tvm_tirx          # Target, runtime tensors, low-level IR/transform
from tvm import relax
```

This is **not** the legacy Relay/AutoTVM stack; don't install TVM v0.19 for this project.

## Recommended: Google Colab NVIDIA GPU

1. Open [`notebooks/relax_metaschedule_colab.ipynb`](../notebooks/relax_metaschedule_colab.ipynb) via the README's Colab link.
2. Select **Runtime → Change runtime type → T4 GPU** (or another NVIDIA GPU).
3. Run the preflight cell. It prints PyTorch and TVM CUDA availability without inventing a pass.
4. Install the project's editable package and the pinned TVM wheel from the notebook.
5. Train five epochs (or load a checkpoint from a previous trusted LeNet-5 run).
6. Execute the smoke-test MetaSchedule experiment, then optionally the full tuning trial budget.
7. Download the ZIP containing the produced JSON and tuning database.

Free Colab sessions have variable resources, time limits, and availability. MetaSchedule can take longer than regular model inference because every candidate schedule must be compiled and measured.

## Local Linux setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip install 'apache-tvm[cuda,torch,meta-schedule]==0.27.0.post1'
python -m pip install 'apache-tvm-tirx[cuda]==0.1.0'
python -m relax_meta.environment
```

If `pytorch_cuda_available` is `false`, select a CUDA-enabled PyTorch build using <https://pytorch.org/get-started/locally/> and check `nvidia-smi`.

If `tvm_cuda_available` is `false` but PyTorch sees the GPU, this TVM installation may lack CUDA support. Review [Apache TVM installation](https://tvm.apache.org/docs/install/index.html) and [source build](https://tvm.apache.org/docs/install/from_source.html). Do not report a MetaSchedule benchmark result from such an environment.

## Common failures

| Symptom | Likely cause | Remedy |
|---|---|---|
| `No module named tvm` | Wheel not installed in current interpreter | Re-run pinned TVM installation in notebook/kernel |
| `No module named tvm_tirx` | Missing separate TIRx dependency | Install `apache-tvm-tirx[cuda]==0.1.0` explicitly |
| `PyTorch CUDA unavailable` | CPU runtime, incompatible driver or PyTorch build | Attach GPU and verify `torch.cuda.is_available()` |
| `TVM CUDA runtime unavailable` | TVM built without CUDA support | Rebuild/install CUDA-capable TVM |
| Relax frontend export fails | PyTorch/TVM operator or version mismatch | Record version, inspect failed operator; check upstream model import docs |
| No schedule records | Tuning found no valid candidate, budget exhaustion, or driver/compile failure | Inspect `tuning_logs/`, increase budget, verify target and toolchain |
| Numerical validation fails | Unsupported operator semantics or scheduling bug | Stop and investigate; do **not** publish a speedup |
| GPU session disconnects | MetaSchedule compile/search exceeds free session limits | Start with one batch and a 16–32 trial budget; persist logs |

## Sources

- <https://tvm.apache.org/docs/how_to/tutorials/e2e_opt_model.html>
- <https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html>
- <https://tvm.apache.org/docs/get_started/tutorials/quick_start.html>
- <https://pypi.org/project/apache-tvm/>

- <https://tvm.apache.org/docs/tirx/install.html>
- <https://pypi.org/project/apache-tvm-tirx/>
