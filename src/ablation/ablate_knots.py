# src/ablation/ablate_knots.py
"""
Ablation: Number of knots.

Runs quick experiments for:
 - Toy regression (KAN with different n_knots)
 - CIFAR (CNN with KAN head varying n_knots in head -- quick mode)

Saves:
 - results/ablation/knot_ablation_toy.csv
 - results/ablation/knot_ablation_cifar.csv
 - plots: knot_vs_mse_toy.png, knot_vs_acc_cifar.png
"""
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from src.models.mlp import SimpleMLP
from src.models.cnn_with_kan import SimpleCNN
from src.data.toy_regression import ToySinDataset
from src.data.cifar10_loader import get_cifar10_dataloaders

def quick_train_regression(model, loader, epochs=8, device='cpu'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for e in range(epochs):
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
    return model

def quick_train_classification(model, loader, epochs=5, device='cpu'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for e in range(epochs):
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y.long())
            loss.backward()
            opt.step()
    return model

def eval_toy_knots(n_knots_list, save_dir, device='cpu'):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ds = ToySinDataset()
    X, Y = [], []
    for i in range(len(ds)):
        x,y = ds[i]
        X.append(x); Y.append(y)
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y), dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    results = []
    for nkn in n_knots_list:
        print("Toy: training with n_knots=", nkn)
        model = SimpleMLP(input_dim=1, hidden_sizes=[64,64,32], num_classes=1,
                          activation='kan',
                          kan_params={'n_knots': nkn, 'x_min': -3.0, 'x_max': 3.0})
        model = quick_train_regression(model, loader, epochs=10, device=device)

        model.eval()
        with torch.no_grad():
            pred = model(X.to(device))
            mse = nn.MSELoss()(pred, Y.to(device)).item()
        results.append({"n_knots": nkn, "mse": mse})
        # save model for reference
        torch.save(model.state_dict(), os.path.join(save_dir, f"toy_kn{nkn}_model.pth"))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "knot_ablation_toy.csv"), index=False)
    # plot
    plt.figure()
    plt.plot(df["n_knots"], df["mse"], marker='o')
    plt.xlabel("n_knots")
    plt.ylabel("MSE")
    plt.title("Toy: Knot ablation")
    plt.savefig(os.path.join(save_dir, "knot_vs_mse_toy.png"), dpi=200)
    plt.close()
    print("Saved toy knot ablation results to", save_dir)
    return df

def eval_cifar_knots(n_knots_list, save_dir, device='cpu'):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    trainloader, testloader = get_cifar10_dataloaders(batch_size=128, quick=True)
    results = []
    for nkn in n_knots_list:
        print("CIFAR: training KAN head with n_knots=", nkn)
        model = SimpleCNN(num_classes=10, use_kan_head=True, kan_params={'n_knots': nkn, 'x_min': -5.0, 'x_max':5.0})
        model = quick_train_classification(model, trainloader, epochs=6, device=device)

        # eval accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x,y in testloader:
                out = model(x.to(device))
                pred = out.argmax(1)
                correct += (pred.cpu() == y).sum().item()
                total += y.size(0)
        acc = correct/total
        results.append({"n_knots": nkn, "acc": acc})
        torch.save(model.state_dict(), os.path.join(save_dir, f"cifar_kn{nkn}_model.pth"))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "knot_ablation_cifar.csv"), index=False)
    plt.figure()
    plt.plot(df["n_knots"], df["acc"], marker='o')
    plt.xlabel("n_knots")
    plt.ylabel("Accuracy")
    plt.title("CIFAR: Knot ablation (KAN head)")
    plt.savefig(os.path.join(save_dir, "knot_vs_acc_cifar.png"), dpi=200)
    plt.close()
    print("Saved cifar knot ablation results to", save_dir)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", type=str, default="./results/ablation")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--knots", type=int, nargs="+", default=[11,21,41])
    args = p.parse_args()
    n_knots_list = args.knots
    save_dir = args.save_dir
    device = args.device

    print("Running knot ablations:", n_knots_list)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    eval_toy_knots(n_knots_list, save_dir, device=device)
    eval_cifar_knots(n_knots_list, save_dir, device=device)

if __name__ == "__main__":
    main()
