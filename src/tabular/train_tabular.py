# src/tabular/train_tabular.py
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.ensemble import RandomForestRegressor
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from src.tabular.load_uci import load_housing, load_energy
from src.tabular.metrics import compute_metrics
from src.models.mlp import SimpleMLP
from src.utils.seed import set_seed
from src.utils.utils import save_checkpoint

def get_dataset(name, quick=False):
    if name == "housing":
        X_train, X_test, y_train, y_test, features = load_housing()
    elif name == "energy":
        X_train, X_test, y_train, y_test, features = load_energy()
    else:
        raise ValueError("Dataset must be housing or energy")

    if quick:
        X_train = X_train[:500]
        y_train = y_train[:500]
        X_test = X_test[:200]
        y_test = y_test[:200]

    return X_train, X_test, y_train, y_test, features


def train_mlp(X_train, y_train, X_test, y_test, activation='relu'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    ds = TensorDataset(X_train_t, y_train_t)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = SimpleMLP(
        input_dim=X_train.shape[1],
        hidden_sizes=[128, 64],
        num_classes=1,
        activation=activation,
        kan_params={"n_knots": 21, "x_min": -3, "x_max": 3}
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Train small number of epochs
    for epoch in range(50):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred = model(X_test_t.to(device)).cpu().numpy().flatten()

    return model, pred


def train_rf(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, pred


def train_xgb(X_train, y_train, X_test):
    if not HAS_XGB:
        print("XGBoost not installed, skipping.")
        return None, None
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror"
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, choices=["housing", "energy"], required=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--save_dir", type=str, default="./results/tabular")
    args = p.parse_args()

    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, feat = get_dataset(args.dataset, quick=args.quick)

    results = []

    # ----- RF -----
    rf_model, rf_pred = train_rf(X_train, y_train, X_test)
    rf_metrics = compute_metrics(y_test, rf_pred)
    results.append(("RandomForest", rf_metrics))

    # ----- XGBoost -----
    if HAS_XGB:
        x_model, x_pred = train_xgb(X_train, y_train, X_test)
        x_metrics = compute_metrics(y_test, x_pred)
        results.append(("XGBoost", x_metrics))

    # ----- MLP baseline -----
    mlp_model, mlp_pred = train_mlp(X_train, y_train, X_test, y_test, activation="relu")
    mlp_metrics = compute_metrics(y_test, mlp_pred)
    results.append(("MLP_ReLU", mlp_metrics))

    # ----- KAN-MLP -----
    kan_model, kan_pred = train_mlp(X_train, y_train, X_test, y_test, activation="kan")
    kan_metrics = compute_metrics(y_test, kan_pred)
    results.append(("KAN_MLP", kan_metrics))

    # Save CSV
    csv_path = os.path.join(args.save_dir, f"{args.dataset}_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("model,rmse,mae,r2\n")
        for name, m in results:
            f.write(f"{name},{m['rmse']},{m['mae']},{m['r2']}\n")

    print("\nResults saved to:", csv_path)
    for name, m in results:
        print(name, m)


if __name__ == "__main__":
    main()
