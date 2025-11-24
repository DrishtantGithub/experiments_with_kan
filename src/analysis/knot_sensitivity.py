# src/analysis/knot_sensitivity.py
"""
Knot Sensitivity Analysis.

Saves:
  - results/interpretability/knot_sensitivity/<run_name>/knot_gradients.csv
  - heatmaps per spline module: knot_importance_<module>.png

Usage examples:

Toy model (MSE):
  python -m src.analysis.knot_sensitivity \
    --model ./results/toy_kan/toy_model.pth \
    --model-type toy \
    --dataset toy \
    --save-dir ./results/interpretability/knot_sensitivity/toy

CIFAR model (small subset, CrossEntropy):
  python -m src.analysis.knot_sensitivity \
    --model ./results/cifar_kan/cifar_model.pth \
    --model-type cnn \
    --dataset cifar_quick \
    --save-dir ./results/interpretability/knot_sensitivity/cifar \
    --num-batches 20
"""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.models.kan_layer import SplineActivation

# model builders (import lazily to avoid overhead)
from src.models.mlp import SimpleMLP
from src.models.cnn_with_kan import SimpleCNN

# toy dataset loader (assumes exists)
from src.data.toy_regression import ToySinDataset
from src.data.cifar10_loader import get_cifar10_dataloaders

def safe_load_state(model, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        # try common containers
        if 'model' in state and isinstance(state['model'], dict):
            sd = state['model']
        elif 'state_dict' in state and isinstance(state['state_dict'], dict):
            sd = state['state_dict']
        else:
            sd = state
        try:
            model.load_state_dict(sd)
        except Exception:
            model.load_state_dict(sd, strict=False)
    else:
        try:
            model.load_state_dict(state)
        except Exception:
            model.load_state_dict(state, strict=False)
    return model

def collect_knot_gradients(model, dataloader, loss_fn, device='cpu', max_batches=None):
    """
    For each SplineActivation module, compute gradient norm for each knot (y parameter).
    We'll accumulate absolute gradient sums or L2 norms over batches.

    Returns:
      results: list of dicts with keys:
        module, unit, knot_index, grad_norm
    """
    model.to(device)
    model.eval()

    # Collect spline modules and their y parameters
    spline_modules = []
    for name, m in model.named_modules():
        if isinstance(m, SplineActivation):
            if not hasattr(m, 'y'):
                continue
            # m.y could be (features, n_knots) or (n_knots,)
            spline_modules.append((name, m))

    if len(spline_modules) == 0:
        print("No SplineActivation modules found in model.")
        return []

    # Prepare accumulators: dict module_name -> numpy array shape (units, n_knots)
    accum = {}
    counts = {}
    for name, m in spline_modules:
        y = m.y.detach()
        if y.dim() == 1:
            units = 1
            n_knots = y.shape[0]
        else:
            units = y.shape[0]
            n_knots = y.shape[1]
        accum[name] = np.zeros((units, n_knots), dtype=np.float64)
        counts[name] = 0

    # iterate batches, compute loss and gradients
    batches = 0
    for batch in dataloader:
        if max_batches is not None and batches >= max_batches:
            break
        model.zero_grad()
        # prepare input, target depending on dataloader type
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y_true = batch[0], batch[1]
        else:
            # single array (toy dataset might yield (x, y) though)
            x = batch
            y_true = None

        x = x.to(device)
        if y_true is not None:
            y_true = y_true.to(device)

        # forward
        logits = model(x)
        if logits is None:
            break

        # choose loss
        if isinstance(loss_fn, nn.MSELoss):
            # regression -> logits shape (B,1) or (B,)
            target = y_true.view_as(logits).to(device)
            loss = loss_fn(logits, target)
        else:
            # classification
            if y_true is None:
                # skip if no labels
                continue
            # logits shape (B, C)
            loss = loss_fn(logits, y_true.long().to(device))

        # backward
        loss.backward()

        # for each spline module, read parameter gradients
        for name, m in spline_modules:
            param = m.y  # tensor
            if param.grad is None:
                # zero gradient: count but continue
                grad_np = np.zeros_like(accum[name])
            else:
                g = param.grad.detach().cpu().numpy()
                # ensure shape (units, n_knots)
                if g.ndim == 1:
                    g = g[None, :]
                # take absolute or l2 per knot
                grad_np = np.abs(g)  # shape matches accum[name]
            accum[name] += grad_np
            # clear grad for safety
            param.grad.zero_()
        batches += 1
        counts[name] += 1  # counts per last module is fine (we'll use batches count)

    # normalize by number of batches processed (batches)
    for name in accum.keys():
        if batches > 0:
            accum[name] = accum[name] / float(batches)

    # prepare results list
    results = []
    for name, arr in accum.items():
        units, n_knots = arr.shape
        for u in range(units):
            for k in range(n_knots):
                results.append({
                    "module": name,
                    "unit": int(u),
                    "knot_index": int(k),
                    "grad_norm": float(arr[u, k])
                })
    return results

def make_dataloader_for_toy(batch_size=256, max_samples=2000, device='cpu'):
    """
    Load the toy regression dataset (ToySinDataset) and take the first max_samples entries.
    This works even if ToySinDataset does NOT take num_samples as an argument.
    """
    # Try instantiating without arguments
    try:
        ds = ToySinDataset()
    except TypeError:
        # If your class requires a different arg (unlikely), fallback to 2000 sample default
        ds = ToySinDataset

    # Extract first max_samples samples
    xs = []
    ys = []
    for i in range(min(max_samples, len(ds))):
        x, y = ds[i]
        xs.append(x)
        ys.append(y)

    X = torch.tensor(np.stack(xs, axis=0), dtype=torch.float32)
    Y = torch.tensor(np.stack(ys, axis=0), dtype=torch.float32)
    ds_t = torch.utils.data.TensorDataset(X, Y)

    loader = torch.utils.data.DataLoader(ds_t, batch_size=batch_size, shuffle=True)
    return loader


def make_dataloader_for_cifar(batch_size=64, quick=True):
    # get test loader from data loader helper
    trainloader, testloader = get_cifar10_dataloaders(batch_size=batch_size, quick=quick)
    # choose testloader
    return testloader

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--model-type", type=str, choices=["toy","cnn"], default="toy")
    p.add_argument("--dataset", type=str, choices=["toy","cifar_quick"], default="toy")
    p.add_argument("--save-dir", type=str, default="./results/interpretability/knot_sensitivity")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num-batches", type=int, default=50, help="max batches to use (CIFAR)")
    args = p.parse_args()

    device = args.device

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Build model
    if args.model_type == "toy":
        model = SimpleMLP(input_dim=1, hidden_sizes=[64,64,32], num_classes=1,
                          activation='kan', kan_params={'n_knots':21,'x_min':-3.0,'x_max':3.0})
    else:
        model = SimpleCNN(num_classes=10, use_kan_head=True)

    model = safe_load_state(model, args.model)
    model.to(device)

    # Prepare dataloader and loss
    if args.dataset == "toy":
        dataloader = make_dataloader_for_toy(batch_size=256, max_samples=2000, device=device)
        loss_fn = nn.MSELoss()
        max_batches = None
    else:
        dataloader = make_dataloader_for_cifar(batch_size=64, quick=True)
        loss_fn = nn.CrossEntropyLoss()
        max_batches = args.num_batches

    # Ensure model parameters require grad for spline y
    for name, m in model.named_modules():
        if isinstance(m, SplineActivation) and hasattr(m, 'y'):
            m.y.requires_grad = True

    print("Collecting knot gradients (this may take a short while)...")
    results = collect_knot_gradients(model, dataloader, loss_fn, device=device, max_batches=max_batches)

    if len(results) == 0:
        print("No results collected.")
        return

    # save CSV
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.save_dir, "knot_gradients.csv")
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    # plot heatmaps per module: reshape
    modules = df["module"].unique()
    for mod in modules:
        dfm = df[df["module"] == mod]
        units = int(dfm["unit"].max()) + 1
        knots = int(dfm["knot_index"].max()) + 1
        arr = np.zeros((units, knots), dtype=np.float32)
        for _, row in dfm.iterrows():
            arr[int(row["unit"]), int(row["knot_index"])] = float(row["grad_norm"])

        plt.figure(figsize=(8, max(2, units/4)))
        plt.imshow(arr, aspect='auto', interpolation='nearest')
        plt.colorbar(label='avg |grad| per knot')
        plt.xlabel('knot index')
        plt.ylabel('unit')
        plt.title(f'Knot gradient importance - {mod}')
        out = os.path.join(args.save_dir, f'knot_importance_{mod.replace(".", "_")}.png')
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
        print("Saved heatmap:", out)

    print("Knot sensitivity analysis complete. Results saved to:", args.save_dir)

if __name__ == "__main__":
    main()
