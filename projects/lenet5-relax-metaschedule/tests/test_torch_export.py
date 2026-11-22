"""Ensure the source CNN exports to a graph that Relax can import."""
import torch
import pytest
from relax_meta.model import LeNet5


@pytest.mark.parametrize("batch_size", [1, 128])
def test_static_torch_export(batch_size):
    model = LeNet5().eval()
    sample = torch.randn(batch_size, 1, 28, 28)
    exported = torch.export.export(model, (sample,), strict=True)
    assert exported is not None
    assert len(list(exported.graph.nodes)) > 0
