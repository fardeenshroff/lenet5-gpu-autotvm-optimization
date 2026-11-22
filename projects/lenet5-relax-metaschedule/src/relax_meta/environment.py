"""Print a truthful preflight check without requiring an available GPU."""
from __future__ import annotations

import json
import platform
import shutil


def main() -> None:
    import torch
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pytorch_cuda_available": bool(torch.cuda.is_available()),
        "nvidia_smi_available": bool(shutil.which("nvidia-smi")),
    }
    if report["pytorch_cuda_available"]:
        report["pytorch_gpu"] = torch.cuda.get_device_name(0)
    try:
        import tvm
        import tvm.tirx
        tvm_tirx = tvm
        from tvm import relax
        from tvm.s_tir.meta_schedule import relax_integration
        report["tvm"] = getattr(tvm, "__version__", "unknown")
        report["relax_available"] = bool(relax)
        report["metaschedule_available"] = bool(relax_integration)
        report["tvm_cuda_available"] = bool(tvm_tirx.cuda(0).exist)
    except (ImportError, AttributeError, RuntimeError) as error:
        report["tvm_error"] = str(error)
        report["tvm_cuda_available"] = False
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
