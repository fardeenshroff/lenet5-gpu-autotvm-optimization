# Generated benchmark artifacts

This directory initially contains **no compiler benchmark results**.

Run `lenet-relax-benchmark` with a functioning NVIDIA CUDA + Apache TVM environment. The command writes actual JSON measurements only after default/tuned Relax outputs pass numerical checks against PyTorch. Keep `tuning_logs/` separately to preserve the MetaSchedule database and audit schedule-search trials.

Do not copy the earlier project's 18.03× PyTorch CPU/GPU speedup into the new MetaSchedule results table: they measure different hypotheses.
