"""Train a local MNIST reference checkpoint; this is not MetaSchedule tuning."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import LeNet5
from .report import save_json


@torch.no_grad()
def test_accuracy(model: LeNet5, batches: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for data, target in batches:
        predicted = model(data.to(device)).argmax(dim=1)
        correct += int((predicted == target.to(device)).sum().item())
        total += len(target)
    return correct / total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--checkpoint", default="checkpoints/lenet5_5epochs.pt")
    parser.add_argument("--output", default="results/training.json")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be positive")
    dev_str = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    dev_str = "cpu" if dev_str == "auto" else dev_str
    if dev_str == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Choose Colab GPU runtime or --device cpu")
    device = torch.device(dev_str)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=norm)
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=norm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512)
    model = LeNet5().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    epochs = []
    overall_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        loss_total = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        accuracy = test_accuracy(model, test_loader, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = {"epoch": epoch, "mean_train_loss": loss_total / len(train_loader),
                  "test_accuracy": accuracy, "wall_seconds": time.perf_counter() - started}
        epochs.append(record)
        print(json.dumps(record), flush=True)
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.cpu().state_dict(), "dropout": True,
                "epochs": args.epochs, "seed": args.seed}, checkpoint)
    report = {"backend": "PyTorch Adam training", "checkpoint": str(checkpoint),
              "epochs": epochs, "device": str(device), "batch_size": args.batch_size,
              "final_test_accuracy": epochs[-1]["test_accuracy"],
              "wall_seconds": time.perf_counter() - overall_start,
              "torch_version": torch.__version__}
    save_json(args.output, report)
    print(f"Saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
