# src/robustness/noise_robustness.py
"""
Noise Robustness Evaluation for:
  - Toy Regression (KAN vs MLP baseline)
  - CIFAR-10 (CNN vs CNN+KAN head)

Outputs:
  noise_metrics_toy.csv
  noise_metrics_cifar.csv
  toy_noise_plot.png
  cifar_noise_plot.png

Usage:
  python -m src.robustness.noise_robustness \
      --toy-kan ./results/toy_kan/toy_model.pth \
      --toy-mlp ./results/toy_relu/toy_model.pth \
      --cifar-kan ./results/cifar_kan/cifar_model.pth \
      --cifar-mlp ./results/cifar_relu/cifar_model.pth \
      --save-dir ./results/robustness
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# model imports
from src.models.mlp import SimpleMLP
from src.models.cnn_with_kan import SimpleCNN

# data imports
from src.data.toy_regression import ToySinDataset
from src.data.cifar10_loader import get_cifar10_dataloaders

def safe_load(model, path):
    """Loads state dict safely."""
    state = torch.load(path, map_location="cpu")
    try:
        model.load_state_dict(state)
    except:
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    return model


# --------------------------------------------------------
# TOY NOISE ROBUSTNESS
# --------------------------------------------------------
def eval_toy_noise(kan_model_path, mlp_model_path, sigmas, device="cpu"):

    # load models
    kan = SimpleMLP(1, [64,64,32], 1, activation="kan",
                    kan_params={"n_knots":21, "x_min":-3, "x_max":3})
    mlp = SimpleMLP(1, [64,64,32], 1, activation="relu")

    kan = safe_load(kan, kan_model_path).to(device)
    mlp = safe_load(mlp, mlp_model_path).to(device)

    ds = ToySinDataset()   # full dataset
    X = []
    Y = []
    for i in range(len(ds)):
        x, y = ds[i]
        X.append(x)
        Y.append(y)
    X = torch.tensor(np.stack(X), dtype=torch.float32).to(device)
    Y = torch.tensor(np.stack(Y), dtype=torch.float32).to(device)

    mse = nn.MSELoss()
    results = []

    for sigma in sigmas:
        noise = torch.randn_like(X) * sigma
        X_noisy = X + noise

        with torch.no_grad():
            pred_kan = kan(X_noisy)
            pred_mlp = mlp(X_noisy)

        loss_kan = mse(pred_kan, Y).item()
        loss_mlp = mse(pred_mlp, Y).item()

        results.append({
            "sigma": sigma,
            "KAN_MSE": loss_kan,
            "MLP_MSE": loss_mlp
        })
        print(f"[Toy] sigma={sigma}: KAN={loss_kan:.4f}, MLP={loss_mlp:.4f}")

    return results


# --------------------------------------------------------
# CIFAR NOISE ROBUSTNESS
# --------------------------------------------------------
def eval_cifar_noise(kan_model_path, mlp_model_path, sigmas, device="cpu"):

    # load loaders
    _, testloader = get_cifar10_dataloaders(batch_size=64, quick=True)

    # load models
    kan = SimpleCNN(num_classes=10, use_kan_head=True)
    mlp = SimpleCNN(num_classes=10, use_kan_head=False)

    kan = safe_load(kan, kan_model_path).to(device)
    mlp = safe_load(mlp, mlp_model_path).to(device)

    ce = nn.CrossEntropyLoss()
    results = []

    for sigma in sigmas:
        total_kan = 0
        total_mlp = 0
        correct_kan = 0
        correct_mlp = 0

        for x, y in testloader:
            x = x.to(device)
            y = y.to(device)

            noise = torch.randn_like(x) * sigma
            x_noisy = x + noise

            with torch.no_grad():
                out_kan = kan(x_noisy)
                out_mlp = mlp(x_noisy)

            # accuracy
            pred_kan = out_kan.argmax(1)
            pred_mlp = out_mlp.argmax(1)

            correct_kan += (pred_kan == y).sum().item()
            correct_mlp += (pred_mlp == y).sum().item()

            total_kan += y.size(0)
            total_mlp += y.size(0)

        acc_kan = correct_kan / total_kan
        acc_mlp = correct_mlp / total_mlp

        results.append({
            "sigma": sigma,
            "KAN_acc": acc_kan,
            "CNN_acc": acc_mlp
        })
        print(f"[CIFAR] sigma={sigma}: KAN_acc={acc_kan:.4f}, CNN_acc={acc_mlp:.4f}")

    return results


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--toy-kan", type=str, required=True)
    p.add_argument("--toy-mlp", type=str, required=True)
    p.add_argument("--cifar-kan", type=str, required=True)
    p.add_argument("--cifar-mlp", type=str, required=True)
    p.add_argument("--save-dir", type=str, default="./results/robustness")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    sigmas = [0.0, 0.1, 0.3, 0.5, 1.0]

    # --- Toy ---
    toy_results = eval_toy_noise(args.toy_kan, args.toy_mlp, sigmas, device=args.device)

    df_toy = pd.DataFrame(toy_results)
    df_toy.to_csv(os.path.join(args.save_dir, "noise_metrics_toy.csv"), index=False)

    # plot
    plt.figure()
    plt.plot(df_toy["sigma"], df_toy["KAN_MSE"], marker="o", label="KAN MSE")
    plt.plot(df_toy["sigma"], df_toy["MLP_MSE"], marker="o", label="MLP MSE")
    plt.xlabel("Noise sigma")
    plt.ylabel("MSE")
    plt.title("Toy Noise Robustness")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, "toy_noise_plot.png"), dpi=200)
    plt.close()

    # --- CIFAR ---
    cifar_results = eval_cifar_noise(args.cifar_kan, args.cifar_mlp, sigmas, device=args.device)

    df_cifar = pd.DataFrame(cifar_results)
    df_cifar.to_csv(os.path.join(args.save_dir, "noise_metrics_cifar.csv"), index=False)

    # plot
    plt.figure()
    plt.plot(df_cifar["sigma"], df_cifar["KAN_acc"], marker="o", label="CNN+KAN Acc")
    plt.plot(df_cifar["sigma"], df_cifar["CNN_acc"], marker="o", label="CNN Acc")
    plt.xlabel("Noise sigma")
    plt.ylabel("Accuracy")
    plt.title("CIFAR Noise Robustness")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, "cifar_noise_plot.png"), dpi=200)
    plt.close()

    print("Noise robustness evaluation saved in:", args.save_dir)


if __name__ == "__main__":
    main()
