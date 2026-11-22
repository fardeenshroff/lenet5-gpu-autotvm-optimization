# Experimental design: isolating compiler tuning from hardware acceleration

## Objective and hypotheses

**Research question:** For a fixed LeNet-5 inference workload running on one NVIDIA GPU, how much latency improvement is attributable to MetaSchedule's hardware-driven search relative to TVM Relax's default GPU scheduling?

- **H0:** The median tuned/default Relax latency ratio is at least 1 (no tuning benefit).
- **H1:** The median tuned/default Relax latency ratio is less than 1 (tuning improves latency).

These are descriptive hypotheses. Default timing uses only seven repeated samples in one session and does **not** constitute a formal hypothesis test.

## Independent and controlled variables

| Dimension | Requirement |
|---|---|
| Independent variable | TIR schedule: default GPU schedule vs MetaSchedule-tuned schedule |
| Models | Identical LeNet-5 architecture and trained state dictionary |
| Hardware | Identical NVIDIA GPU and driver throughout the paired comparison |
| Shapes | Separate fixed-shape experiments at N=1 and N=128 |
| Dtype and layout | FP32, input NCHW (N,1,28,28) |
| Software | Same TVM/PyTorch versions and target for all variants |
| Synchronization | Warm up once, synchronize before/after each measurement group |
| Timing | Group average latency per invocation; median of repeated groups |
| Transfers | Exclude initial H2D and output D2H for **all** model backends |
| Compilation | Exclude compilation and tuning from inference time; report tuning time separately |
| Correctness | Every compiler output checked against PyTorch with rtol=atol=1e-3 |

## Compiler pathway

1. Deserialize a trusted PyTorch checkpoint; set `model.eval()` to disable dropout.
2. Use `torch.export.export` to capture a fixed batch-size inference graph.
3. Convert the exported graph with `tvm.relax.frontend.torch.from_exported_program(..., keep_params_as_input=True)`, then `detach_params()`.
4. Lower to TIR via the `zero` Relax pipeline.
5. Compile a control VM using `DefaultGPUSchedule` applied to the legalized IR.
6. Independently run `MetaScheduleTuneTIR`, `MetaScheduleApplyDatabase`, and fallback GPU scheduling on the same raw Relax IR.
7. Require at least one persisted tuning record; otherwise abort with no tuned result.
8. Validate default and tuned outputs against PyTorch (and against each other) before collecting performance.
9. Benchmark each version using exactly the same GPU-resident tensors and the same synchronization protocol.
10. Write a JSON record only after **all selected batch sizes** complete successfully.

## Timing semantics

For each of 7 samples, issue 50 forward calls and synchronize at the end. The recorded value is the aggregate elapsed time divided by the number of calls. The median over samples is used to avoid reliance on the most favorable sample.

This measures device computation plus framework dispatch/VM overhead, but it excludes parameter upload and output transfer. If a later study aims to measure application-level latency, a *separate* end-to-end benchmark must include those transfers, preprocessing, and postprocessing.

## MetaSchedule database

Tuning is separated by batch shape (`tuning_logs/batch_1/`, `tuning_logs/batch_128/`). The expected persistent files are `database_workload.json` and `database_tuning_record.json`. The benchmark reports records present before tuning, after tuning, and newly written. A nonempty database verifies recorded search activity; it does **not** guarantee a speedup or prove that every graph operation received tuning coverage.

## Limitations and improvements

- TVM includes graph-level optimization and kernel fusion. The *B versus C* comparison controls for baseline graph import and legalization, but MetaSchedule may affect the shape of downstream scheduled kernels.
- A GPU that is thermally or resource constrained can distort timings. Record the target clock and GPU utilization if available, and execute repeated independent experiments.
- Multiple batch sizes should be analyzed separately; 128-sized results cannot be generalized to real-time batch-1 deployment.
- A compilation error or numerical mismatch invalidates the corresponding run. Never fill missing metrics from PyTorch-only data or an older AutoTVM report.
- Training and tuning are **not the same optimization problem**. The present benchmark tunes inference TIR workloads, not gradient/backpropagation kernels.
- For stronger conclusions run 5–10 independent sessions per configuration, report median and 95% confidence intervals, record trial budgets, and compare `trials=0`, 64, 128, 512 under a fixed tuning budget.
