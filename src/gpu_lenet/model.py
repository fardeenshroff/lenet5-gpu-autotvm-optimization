"""LeNet-5 for MNIST, with two convolution and three dense layers."""
from torch import nn


class LeNet5(nn.Module):
    """28x28 grayscale image input, ten-class output."""

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

    def forward(self, x):
        return self.classifier(self.features(x))
