# src/robustness/low_data.py
"""
Low-Data Robustness Evaluation for:
  - Toy Regression (KAN vs MLP baseline)
  - CIFAR-10 (CNN vs CNN+KAN head)

Trains on fractions {0.1, 0.2, 0.5, 1.0} of data.

Outputs:
  lowdata_toy.csv
  lowdata_cifar.csv
  toy_lowdata_plot.png
  cifar_lowdata_plot.png

Usage:
  python -m src.robustness.low_data \
      --toy-kan ./results/toy_kan/toy_model.pth \
      --toy-mlp ./results/toy_relu/toy_model.pth \
      --cifar-kan ./results/cifar_kan/cifar_model.pth \
      --cifar-mlp ./results/cifar_relu/cifar_model.pth \
      --save-dir ./results/robustness_lowdata
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt

# Model imports
from src.models.mlp import SimpleMLP
from src.models.cnn_with_kan import SimpleCNN

# Data imports
from src.data.toy_regression import ToySinDataset
from src.data.cifar10_loader import get_cifar10_dataloaders


# ----------------------------------------------
# Helper function to train a simple model quickly
# ----------------------------------------------
def quick_train(model, loader, criterion, epochs=3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
    return model


# ----------------------------------------------------
# Low-data robustness for TOY regression
# ----------------------------------------------------
def lowdata_toy(kan_model_path, mlp_model_path, device="cpu"):

    # Load full dataset
    ds = ToySinDataset()
    X, Y = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        X.append(x)
        Y.append(y)
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y), dtype=torch.float32)

    full_ds = TensorDataset(X, Y)

    fractions = [0.1, 0.2, 0.5, 1.0]
    results = []
    mse = nn.MSELoss()

    for frac in fractions:

        n = int(frac * len(full_ds))
        idx = np.random.choice(len(full_ds), n, replace=False)
        sub = Subset(full_ds, idx)
        loader = DataLoader(sub, batch_size=64, shuffle=True)

        # Fresh models for training
        model_kan = SimpleMLP(1, [64,64,32], 1, activation="kan",
                              kan_params={"n_knots":21, "x_min":-3, "x_max":3})
        model_mlp = SimpleMLP(1, [64,64,32], 1, activation="relu")

        # Train
        quick_train(model_kan, loader, mse, epochs=5, device=device)
        quick_train(model_mlp, loader, mse, epochs=5, device=device)

        # Final error on FULL dataset
        with torch.no_grad():
            pred_kan = model_kan(X.to(device))
            pred_mlp = model_mlp(X.to(device))
        mse_kan = mse(pred_kan, Y.to(device)).item()
        mse_mlp = mse(pred_mlp, Y.to(device)).item()

        results.append({
            "fraction": frac,
            "KAN_MSE": mse_kan,
            "MLP_MSE": mse_mlp
        })

        print(f"[Toy Low-Data] frac={frac}: KAN={mse_kan:.4f}, MLP={mse_mlp:.4f}")

    return results


# ----------------------------------------------------
# Low-data robustness for CIFAR-10
# ----------------------------------------------------
def lowdata_cifar(kan_path, cnn_path, device="cpu"):

    trainloader, testloader = get_cifar10_dataloaders(batch_size=64, quick=True)

    # Get full train dataset from loader
    full_ds = trainloader.dataset
    length = len(full_ds)
    fractions = [0.1, 0.2, 0.5, 1.0]
    results = []

    ce = nn.CrossEntropyLoss()

    for frac in fractions:

        n = int(frac * length)
        idx = np.random.choice(length, n, replace=False)
        sub = Subset(full_ds, idx)
        loader = DataLoader(sub, batch_size=64, shuffle=True)

        # Create fresh models
        model_kan = SimpleCNN(num_classes=10, use_kan_head=True)
        model_cnn = SimpleCNN(num_classes=10, use_kan_head=False)

        # Train
        quick_train(model_kan, loader, ce, epochs=3, device=device)
        quick_train(model_cnn, loader, ce, epochs=3, device=device)

        # Evaluate
        correct_kan = 0
        correct_cnn = 0
        total = 0

        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)

                out_kan = model_kan(x)
                out_cnn = model_cnn(x)

                pred_kan = out_kan.argmax(1)
                pred_cnn = out_cnn.argmax(1)

                correct_kan += (pred_kan == y).sum().item()
                correct_cnn += (pred_cnn == y).sum().item()
                total += y.size(0)

        acc_kan = correct_kan / total
        acc_cnn = correct_cnn / total

        results.append({
            "fraction": frac,
            "KAN_acc": acc_kan,
            "CNN_acc": acc_cnn
        })

        print(f"[CIFAR Low-Data] frac={frac}: KAN_acc={acc_kan:.4f}, CNN_acc={acc_cnn:.4f}")

    return results


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--toy-kan", type=str, required=True)
    p.add_argument("--toy-mlp", type=str, required=True)
    p.add_argument("--cifar-kan", type=str, required=True)
    p.add_argument("--cifar-mlp", type=str, required=True)
    p.add_argument("--save-dir", type=str, default="./results/robustness_lowdata")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Toy
    toy_results = lowdata_toy(args.toy_kan, args.toy_mlp, device=args.device)
    df_toy = pd.DataFrame(toy_results)
    df_toy.to_csv(os.path.join(args.save_dir, "lowdata_toy.csv"), index=False)

    plt.figure()
    plt.plot(df_toy["fraction"], df_toy["KAN_MSE"], marker="o", label="KAN")
    plt.plot(df_toy["fraction"], df_toy["MLP_MSE"], marker="o", label="MLP")
    plt.xlabel("Fraction of Training Data")
    plt.ylabel("MSE")
    plt.title("Toy Low-Data Robustness")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, "toy_lowdata_plot.png"), dpi=200)
    plt.close()

    # CIFAR
    cifar_results = lowdata_cifar(args.cifar_kan, args.cifar_mlp, device=args.device)
    df_cifar = pd.DataFrame(cifar_results)
    df_cifar.to_csv(os.path.join(args.save_dir, "lowdata_cifar.csv"), index=False)

    plt.figure()
    plt.plot(df_cifar["fraction"], df_cifar["KAN_acc"], marker="o", label="CNN+KAN")
    plt.plot(df_cifar["fraction"], df_cifar["CNN_acc"], marker="o", label="CNN")
    plt.xlabel("Fraction of Training Data")
    plt.ylabel("Accuracy")
    plt.title("CIFAR Low-Data Robustness")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, "cifar_lowdata_plot.png"), dpi=200)
    plt.close()

    print("Low-data robustness results saved in:", args.save_dir)


if __name__ == "__main__":
    main()
