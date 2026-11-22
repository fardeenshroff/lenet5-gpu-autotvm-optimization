#!/usr/bin/env python3
"""Legacy TVM Relay CUDA inference and optional AutoTVM operator tuning.

Designed for Apache TVM v0.19.0 built with USE_CUDA=ON. TVM's modern
releases have removed the legacy Relay/AutoTVM APIs used by the study.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch
from gpu_lenet.checkpoint import load_model


def require_tvm():
    try:
        import tvm
        from tvm import autotvm, relay
        from tvm.contrib import graph_executor
    except (ImportError, AttributeError) as exc:
        raise SystemExit("Requires legacy Apache TVM v0.19.0 with CUDA. "
                         "See docs/SETUP.md") from exc
    return tvm, autotvm, relay, graph_executor


def convert(model, batch_size, relay):
    model.eval().cpu()
    sample = torch.randn(batch_size, 1, 28, 28)
    with torch.inference_mode():
        traced = torch.jit.trace(model, sample)
    mod, params = relay.frontend.from_pytorch(traced, [("input0", tuple(sample.shape))])
    return mod, params, sample


def tune(mod, params, target, relay, autotvm, trials, log_path):
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params,
        ops=(relay.op.get("nn.conv2d"),)
    )
    if not tasks:
        print("No tunable convolution tasks found; using default schedules")
        return False
    from tvm.autotvm.tuner import XGBTuner, RandomTuner
    log_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = log_path.with_suffix(".tmp.log")
    temporary.unlink(missing_ok=True)
    options = autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=60),
        runner=autotvm.LocalRunner(number=10, repeat=1, timeout=15,
                                  min_repeat_ms=150),
    )
    for index, task in enumerate(reversed(tasks), start=1):
        n = min(trials, len(task.config_space))
        print(f"Tuning {index}/{len(tasks)}: {n} configurations")
        try:
            tuner = XGBTuner(task, loss_type="rank")
        except Exception:
            tuner = RandomTuner(task)
        tuner.tune(
            n_trial=n, measure_option=options,
            callbacks=[autotvm.callback.progress_bar(n),
                       autotvm.callback.log_to_file(str(temporary))],
        )
    autotvm.record.pick_best(str(temporary), str(log_path))
    temporary.unlink(missing_ok=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/lenet5.pt")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--log", default="results/autotvm_cuda.log")
    p.add_argument("--output", default="results/tvm_cuda_inference.json")
    p.add_argument("--number", type=int, default=10)
    p.add_argument("--repeat", type=int, default=10)
    args = p.parse_args()
    if min(args.batch_size, args.trials, args.number, args.repeat) < 1:
        p.error("All numeric settings must be positive")
    tvm, autotvm, relay, graph_executor = require_tvm()
    device = tvm.cuda(0)
    if not device.exist:
        p.error("TVM cannot access a CUDA device")
    model = load_model(args.checkpoint)
    mod, params, sample = convert(model, args.batch_size, relay)
    target = tvm.target.Target("cuda")
    log_path = Path(args.log)
    if args.tune:
        tune(mod, params, target, relay, autotvm, args.trials, log_path)
    context = autotvm.apply_history_best(str(log_path)) if log_path.exists() else nullcontext()
    with context:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
    runtime = graph_executor.GraphModule(lib["default"](device))
    runtime.set_input("input0", tvm.nd.array(sample.numpy(), device=device))
    runtime.run()
    observed = runtime.get_output(0).numpy()
    with torch.inference_mode():
        expected = model(sample).numpy()
    np.testing.assert_allclose(observed, expected, rtol=1e-3, atol=1e-3)
    bench = runtime.benchmark(device, number=args.number, repeat=args.repeat)
    measured_ms = [float(sec) * 1000 for sec in bench.results]
    report = {
        "backend": "TVM Relay CUDA inference",
        "tuned": log_path.exists(), "batch_size": args.batch_size,
        "tvm_version": getattr(tvm, "__version__", "unknown"),
        "mean_ms": float(np.mean(measured_ms)),
        "median_ms": float(np.median(measured_ms)),
        "min_ms": float(np.min(measured_ms)),
        "max_ms": float(np.max(measured_ms)),
        "correctness": "matched PyTorch output within atol=1e-3, rtol=1e-3",
        "note": "inference only; not historical training benchmark",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
