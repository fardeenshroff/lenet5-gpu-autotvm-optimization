import torch
from gpu_lenet import LeNet5


def test_output_shape():
    model = LeNet5().eval()
    with torch.inference_mode():
        result = model(torch.randn(3, 1, 28, 28))
    assert result.shape == (3, 10)
    assert torch.isfinite(result).all()


def test_eval_deterministic():
    model = LeNet5().eval()
    data = torch.randn(2, 1, 28, 28)
    with torch.inference_mode():
        assert torch.equal(model(data), model(data))


def test_backpropagation():
    model = LeNet5().train()
    model(torch.randn(2, 1, 28, 28)).square().mean().backward()
    assert model.features[0].weight.grad is not None
