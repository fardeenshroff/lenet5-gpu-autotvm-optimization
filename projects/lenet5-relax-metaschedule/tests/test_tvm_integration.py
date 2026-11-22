"""Optional CPU-side smoke for the actual modern TVM frontend APIs.

Set RUN_TVM_CPU_INTEGRATION=1 after installing apache-tvm==0.27.0.post1.
This never claims CUDA or MetaSchedule performance; it only validates
PyTorch export -> Relax import -> legalize -> TIR task extraction.
"""
import os

import pytest
import torch
from relax_meta.model import LeNet5


@pytest.mark.skipif(os.environ.get('RUN_TVM_CPU_INTEGRATION') != '1', reason='opt-in TVM API check')
def test_relax_frontend_and_legalization():
    import tvm
    import tvm_tirx
    from tvm import relax
    from tvm.relax.frontend.torch import from_exported_program
    from tvm.s_tir.meta_schedule.relax_integration import extract_tasks
    from relax_meta.benchmark import import_relax, lower_relax

    model = LeNet5().eval()
    example = torch.randn(1, 1, 28, 28)
    mod, params = import_relax(model, example, relax, from_exported_program)
    assert mod is not None
    assert params
    target = tvm_tirx.target.Target('llvm')
    legalized = lower_relax(mod, relax, target)
    tasks = extract_tasks(legalized, target)
    assert len(tasks) > 0, 'No TIR tasks generated from LeNet-5 Relax graph'
