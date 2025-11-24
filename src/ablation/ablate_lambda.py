# src/ablation/ablate_lambda.py
"""
Ablation: Spline smoothness regularization (lambda) for Toy KAN.

We add a penalty term: lambda * sum_k (second_diff(y_k)^2) across knots and units.

Saves:
 - results/ablation/lambda_ablation_toy.csv
 - results/ablation/curvature_vs_lambda.png
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
from src.data.toy_regression import ToySinDataset

def spline_smoothness_penalty(model):
    """
    Compute sum over all SplineActivation modules of the squared second difference of y.
    Works even if spline is per-channel or shared.
    """
    total = 0.0
    found = False
    for name, m in model.named_modules():
        # lazy import to avoid circular issues
        from src.models.kan_layer import SplineActivation
        if isinstance(m, SplineActivation) and hasattr(m, 'y'):
            found = True
            y = m.y  # tensor
            # shape could be (n_knots,) or (channels, n_knots)
            y_ = y.view(-1, y.shape[-1])  # (units, n_knots)
            # second diff along knot axis
            d2 = y_[:, 2:] - 2*y_[:, 1:-1] + y_[:, :-2]  # shape (units, n_knots-2)
            total = total + (d2.pow(2).sum())
    if not found:
        return torch.tensor(0.0, dtype=torch.float32)
    return total

def quick_train_with_lambda(model, loader, lam, epochs=12, device='cpu'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    model.train()
    for e in range(epochs):
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = mse(pred, y)
            # add spline smoothness penalty
            penalty = spline_smoothness_penalty(model)
            loss = loss + lam * penalty
            loss.backward()
            opt.step()
    return model

def eval_lambda(lambdas, save_dir, device='cpu'):
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
    for lam in lambdas:
        print("Training with lambda=", lam)
        model = SimpleMLP(1, [64,64,32], 1, activation='kan',
                          kan_params={'n_knots':21, 'x_min':-3.0, 'x_max':3.0})
        model = quick_train_with_lambda(model, loader, lam, epochs=12, device=device)
        model.eval()
        with torch.no_grad():
            pred = model(X.to(device))
            mse = nn.MSELoss()(pred, Y.to(device)).item()
        # compute curvature (sum squared second diff) for reporting
        from src.models.kan_layer import SplineActivation
        total_curv = 0.0
        for name, m in model.named_modules():
            if isinstance(m, SplineActivation) and hasattr(m, 'y'):
                y_ = m.y.detach().view(-1, m.y.shape[-1])
                d2 = y_[:, 2:] - 2*y_[:, 1:-1] + y_[:, :-2]
                total_curv += float((d2.pow(2).sum().item()))
        results.append({"lambda": lam, "mse": mse, "curvature": total_curv})
        torch.save(model.state_dict(), os.path.join(save_dir, f"toy_lambda{lam:.0e}_model.pth"))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "lambda_ablation_toy.csv"), index=False)

    # plot
    plt.figure()
    plt.plot(df["lambda"], df["mse"], marker='o')
    plt.xscale('log')
    plt.xlabel("lambda (smoothness penalty)")
    plt.ylabel("MSE")
    plt.title("Toy: lambda ablation (MSE)")
    plt.savefig(os.path.join(save_dir, "curv_vs_lambda_mse.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(df["lambda"], df["curvature"], marker='o')
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("curvature (sum squared second diff)")
    plt.title("Toy: lambda vs curvature")
    plt.savefig(os.path.join(save_dir, "curvature_vs_lambda.png"), dpi=200)
    plt.close()

    print("Saved lambda ablation results to", save_dir)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", type=str, default="./results/ablation/lambda")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1e-4, 1e-3, 1e-2])
    args = p.parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    eval_lambda(args.lambdas, args.save_dir, device=args.device)

if __name__ == "__main__":
    main()
