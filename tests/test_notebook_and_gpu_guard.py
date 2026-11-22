"""Validate the portable GPU notebook and guard against false CUDA claims."""
import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def test_colab_gpu_notebook_code_cells_compile():
    notebook_path = ROOT / "notebooks" / "gpu_benchmark_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"]
                  if cell["cell_type"] == "code"]
    assert len(code_cells) >= 5
    for cell_index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        ast.parse(source, filename=f"{notebook_path}:cell{cell_index}")


def test_gpu_probe_refuses_cpu_only_runtime(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("A real NVIDIA GPU is available; this test covers CPU-only hosts")
    output = tmp_path / "should_not_exist.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "verify_cuda_runtime.py"),
        "--output", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "CUDA not available" in result.stderr
    assert not output.exists()
