#!/usr/bin/env python3
"""Train a reference LeNet-5 checkpoint on MNIST."""
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
from gpu_lenet import LeNet5


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    count = correct = 0
    for data, labels in loader:
        logits = model(data.to(device))
        correct += int((logits.argmax(dim=1) == labels.to(device)).sum().item())
        count += labels.numel()
    return correct / count if count else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="checkpoints/lenet5.pt")
    parser.add_argument("--metrics", default="results/train.json")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    selected = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(selected)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=512)
    model = LeNet5().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    epochs = []
    overall = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        loss_sum = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(data), target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
        if device.type == "cuda":
            torch.cuda.synchronize()
        test_accuracy = accuracy(model, test_loader, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        result = {"epoch": epoch, "mean_train_loss": loss_sum / len(train_loader),
                  "test_accuracy": test_accuracy, "wall_seconds": time.perf_counter() - start}
        epochs.append(result)
        print(json.dumps(result))
    total = time.perf_counter() - overall
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.cpu().state_dict(),
                "dropout": True, "seed": args.seed, "epochs": args.epochs}, path)
    report = {"backend": "PyTorch training", "device": selected,
              "torch_version": torch.__version__, "batch_size": args.batch_size,
              "total_wall_seconds": total, "final_test_accuracy": epochs[-1]["test_accuracy"],
              "epochs": epochs, "checkpoint": str(path)}
    output = Path(args.metrics)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved checkpoint to {path}")


if __name__ == "__main__":
    main()
