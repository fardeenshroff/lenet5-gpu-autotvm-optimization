import torch
from relax_meta.model import LeNet5, load_checkpoint


def test_output_shape_and_finiteness():
    model = LeNet5().eval()
    with torch.inference_mode():
        logits = model(torch.randn(4, 1, 28, 28))
    assert logits.shape == (4, 10)
    assert torch.isfinite(logits).all()


def test_inference_repeatable():
    model = LeNet5().eval()
    x = torch.randn(1, 1, 28, 28)
    with torch.inference_mode():
        assert torch.equal(model(x), model(x))


def test_training_gradients():
    model = LeNet5()
    model(torch.randn(2, 1, 28, 28)).square().mean().backward()
    assert model.features[0].weight.grad is not None


def test_checkpoint_roundtrip(tmp_path):
    model = LeNet5().eval()
    path = tmp_path / "good.pt"
    torch.save({"model_state_dict": model.state_dict(), "dropout": True}, path)
    restored = load_checkpoint(path)
    x = torch.randn(3, 1, 28, 28)
    with torch.inference_mode():
        torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
