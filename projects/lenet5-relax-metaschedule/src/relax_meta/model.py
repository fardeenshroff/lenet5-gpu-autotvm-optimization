"""LeNet-5 variant matching the earlier Colab PyTorch GPU experiment.

This is a 28x28 LeNet-style network rather than a bit-for-bit recreation of
LeCun et al.'s 1998 architecture. Dropout is inactive during inference.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class LeNet5(nn.Module):
    """Input [N, 1, 28, 28]; output [N, 10]."""

    def __init__(self, dropout: bool = True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25 if dropout else 0.0),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Dropout(p=0.5 if dropout else 0.0),
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def load_checkpoint(path: str | Path) -> LeNet5:
    """Load a trusted, locally produced state dictionary; fail on incompatibility."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Train first.")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or "model_state_dict" not in obj:
        raise ValueError("Checkpoint must be a dict with 'model_state_dict'.")
    model = LeNet5(dropout=bool(obj.get("dropout", True)))
    model.load_state_dict(obj["model_state_dict"], strict=True)
    return model.eval().float()
