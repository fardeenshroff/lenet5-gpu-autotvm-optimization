"""Load model weights safely for portable inference experiments."""
from pathlib import Path
import torch
from .model import LeNet5


def load_model(path: str | Path) -> LeNet5:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected checkpoint containing model_state_dict")
    model = LeNet5(dropout=checkpoint.get("dropout", True))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()
