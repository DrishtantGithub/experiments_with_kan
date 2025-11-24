# src/train/train_cifar_residual.py
"""
Train script for SimpleCNNResidual (baseline / KAN / Residual KAN head).

Quick usage (quick mode uses fewer epochs/batches):
  python -m src.train.train_cifar_residual --quick --head linear --save_dir ./results/cifar_baseline
  python -m src.train.train_cifar_residual --quick --head kan --save_dir ./results/cifar_kan
  python -m src.train.train_cifar_residual --quick --head residual_kan --save_dir ./results/cifar_residual_kan
"""
import argparse, os, time
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

from src.train.simple_cnn_residual import SimpleCNNResidual
from src.data.cifar10_loader import get_cifar10_dataloaders

def train(args):
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    trainloader, testloader = get_cifar10_dataloaders(batch_size=args.batch_size, quick=args.quick)

    model = SimpleCNNResidual(num_classes=10, head_type=args.head, kan_params={
        'n_knots': args.n_knots, 'x_min': -5.0, 'x_max': 5.0, 'per_channel': False
    })

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 6 if not args.quick else 3

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            if args.quick and i > 100:  # limit quick mode
                break
        print(f"Epoch {epoch}/{epochs} loss={running_loss/(i+1):.6f}")

        # quick validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in testloader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                pred = out.argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
                if args.quick and total >= 500:
                    break
        acc = correct / total
        print(f"Validation acc={acc:.4f}")

    # save model & metrics
    save_path = os.path.join(args.save_dir, "cifar_model.pth")
    torch.save(model.state_dict(), save_path)
    metrics_path = os.path.join(args.save_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"val_acc={acc:.4f}\nhead={args.head}\n")
    print("Saved model to", save_path)
    return save_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--head", type=str, choices=["linear","kan","residual_kan"], default="linear")
    p.add_argument("--n_knots", type=int, default=21)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train(args)

if __name__ == "__main__":
    main()
