# src/analysis/activation_response.py
"""
Activation response visualization for KAN vs baseline MLP.

Usage examples (toy regression, quick):
  python -m src.analysis.activation_response \
      --kan-model ./results/toy_kan/toy_model.pth \
      --baseline-model ./results/toy_relu/toy_model.pth \
      --model-type toy \
      --save-dir ./results/interpretability/activation_response/toy

Options:
  --model-type: 'toy' (dense x-grid) or 'cnn' (images from cifar test subset)
  --x-min/x-max/x-n: range for toy input grid
  --max-units: limit number of units to plot per layer
"""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.kan_layer import SplineActivation
from src.models.mlp import SimpleMLP
from src.models.cnn_with_kan import SimpleCNN

# safe state loader (tries a few common formats)
def load_state_into_model(model, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and ('state_dict' in state or 'model' in state):
        if 'state_dict' in state:
            sd = state['state_dict']
        elif 'model' in state:
            sd = state['model']
        else:
            sd = state
        try:
            model.load_state_dict(sd)
        except Exception:
            # try non-strict
            model.load_state_dict(sd, strict=False)
    else:
        try:
            model.load_state_dict(state)
        except Exception:
            model.load_state_dict(state, strict=False)
    return model

def register_hooks_for_splines(model, collector):
    """
    Collector will be dict: collector[name] = list of outputs (per forward)
    We'll capture the SplineActivation outputs (the activation values)
    """
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, SplineActivation):
            def make_hook(n):
                def hook(_module, _input, output):
                    # output shape: (..., features) or (...,) ; flatten first dim as sample
                    out_np = output.detach().cpu().numpy()
                    collector.setdefault(n, []).append(out_np)
                return hook
            hooks.append(m.register_forward_hook(make_hook(name)))
    return hooks

def register_hooks_for_mlp_postact(model, collector):
    """
    For baseline MLP (no SplineActivation), capture post-activation values after Linear+ReLU/GELU.
    We'll capture outputs of nn.ReLU or nn.GELU modules if present; else capture outputs of linear layers.
    """
    import torch.nn as nn
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.ReLU, nn.GELU)):
            def make_hook(n):
                def hook(_module, _input, output):
                    collector.setdefault(n, []).append(output.detach().cpu().numpy())
                return hook
            hooks.append(m.register_forward_hook(make_hook(name)))
    # if no activations captured (older SimpleMLP might not expose), fallback to Linear outputs
    if len(hooks) == 0:
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                def make_hook(n):
                    def hook(_module, _input, output):
                        collector.setdefault(n, []).append(output.detach().cpu().numpy())
                    return hook
                hooks.append(m.register_forward_hook(make_hook(name)))
    return hooks

def collect_on_grid(model, model_type, grid_or_loader, batch_transform=None):
    """
    Run model on grid_or_loader and collect activation outputs (via hooks).
    grid_or_loader:
      - for toy: a numpy array shape (N,1)
      - for cnn: a DataLoader or list of (x, y)
    batch_transform: function to convert numpy batch -> torch tensor (device already cpu)
    Returns nothing; hooks fill collectors defined outside.
    """
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    if model_type == "toy":
        X = grid_or_loader  # numpy (N,1)
        batch_size = 256
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
                _ = model(xb)
    else:
        # assume grid_or_loader is an iterable of tensors / (x,y)
        with torch.no_grad():
            for batch in grid_or_loader:
                if isinstance(batch, (list, tuple)):
                    xb = batch[0]
                else:
                    xb = batch
                if isinstance(xb, np.ndarray):
                    xb = torch.tensor(xb, dtype=torch.float32)
                xb = xb.to(device)
                _ = model(xb)

def save_activation_heatmaps(collector, x_vals, save_dir, prefix="kan", max_units=32):
    """
    collector: dict name -> list of outputs per batch (each is array shape (B, features) or (B,))
    For toy, we flatten along sample dimension and build matrix (units x samples).
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for name, outputs in collector.items():
        # stack along samples
        arr = np.concatenate(outputs, axis=0)

        # Handle CNN baseline: reduce spatial dimensions
        if arr.ndim == 4:
            # (N, C, H, W) -> (N, C)
            arr = arr.mean(axis=(2,3))
        elif arr.ndim == 3:
            # (N, C, H) -> (N, C)
            arr = arr.mean(axis=2)

        # Ensure 2D
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim != 2:
            print("Warning: unexpected activation dimension", arr.shape)

        # Transpose: (units x samples)
        arr = arr.T

        n_units = arr.shape[0]
        n_plot = min(n_units, max_units)
        # save CSV
        csv_path = os.path.join(save_dir, f"{prefix}_{name.replace('.', '_')}_activations.csv")
        np.savetxt(csv_path, arr, delimiter=",")
        # heatmap: units x x_vals
        plt.figure(figsize=(8, max(2, n_plot/4)))
        plt.imshow(arr[:n_plot, :], aspect='auto', interpolation='nearest', extent=[0, arr.shape[1], 0, n_plot])
        plt.colorbar(label='activation')
        plt.xlabel('input (toy x) / sample idx')
        plt.ylabel('unit (first axis)')
        plt.title(f"{prefix} activations: {name} (showing {n_plot}/{n_units})")
        out_png = os.path.join(save_dir, f"{prefix}_{name.replace('.', '_')}_heatmap.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        # also plot sample unit line plots for first few units
        for u in range(n_plot):
            plt.figure(figsize=(6,2))
            # automatically match activation length
            num_samples = arr.shape[1]
            x_axis = np.arange(num_samples)  # 0,1,2,...,num_samples-1
            plt.plot(x_axis, arr[u, :], lw=1.5)

            plt.title(f"{prefix} {name} unit {u}")
            plt.xlabel("input (toy x) / sample idx")
            plt.ylabel("activation")
            out_line = os.path.join(save_dir, f"{prefix}_{name.replace('.', '_')}_unit{u}.png")
            plt.tight_layout()
            plt.savefig(out_line, dpi=200)
            plt.close()
        print("Saved activations for", name, "->", csv_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kan-model", type=str, required=True, help="path to KAN model .pth")
    p.add_argument("--baseline-model", type=str, default=None, help="path to baseline MLP model .pth (optional)")
    p.add_argument("--model-type", type=str, choices=["toy","cnn"], default="toy")
    p.add_argument("--save-dir", type=str, default="./results/interpretability/activation_response")
    p.add_argument("--x-min", type=float, default=-3.0)
    p.add_argument("--x-max", type=float, default=3.0)
    p.add_argument("--x-n", type=int, default=800)
    p.add_argument("--max-units", type=int, default=32)
    args = p.parse_args()

    save_dir = args.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if args.model_type == "toy":
        # build toy grid
        xs = np.linspace(args.x_min, args.x_max, args.x_n).reshape(-1,1).astype(np.float32)
        # KAN model (same architecture as training)
        kan = SimpleMLP(input_dim=1, hidden_sizes=[64,64,32], num_classes=1,
                        activation='kan', kan_params={'n_knots':21,'x_min':-3.0,'x_max':3.0})
        kan = load_state_into_model(kan, args.kan_model)

        # baseline MLP
        baseline = None
        if args.baseline_model:
            baseline = SimpleMLP(input_dim=1, hidden_sizes=[64,64,32], num_classes=1,
                                 activation='relu')
            baseline = load_state_into_model(baseline, args.baseline_model)

        # collectors
        kan_col = {}
        base_col = {}

        # register hooks
        kan_hooks = register_hooks_for_splines(kan, kan_col)
        base_hooks = []
        if baseline is not None:
            base_hooks = register_hooks_for_mlp_postact(baseline, base_col)

        # forward on grid
        collect_on_grid(kan, "toy", xs)
        if baseline is not None:
            collect_on_grid(baseline, "toy", xs)

        # remove hooks
        for h in kan_hooks: h.remove()
        for h in base_hooks: h.remove()

        # save heatmaps and csvs
        save_activation_heatmaps(kan_col, xs.squeeze(), os.path.join(save_dir, "kan"), prefix="kan", max_units=args.max_units)
        if baseline is not None:
            save_activation_heatmaps(base_col, xs.squeeze(), os.path.join(save_dir, "baseline"), prefix="baseline", max_units=args.max_units)

        print("Activation response plots saved under:", save_dir)

    else:
        # CIFAR mode: use a small test subset loader to collect responses
        from src.data.cifar10_loader import get_cifar10_dataloaders
        testloader = get_cifar10_dataloaders(batch_size=64, quick=True)[1]
        kan = SimpleCNN(num_classes=10, use_kan_head=True)
        kan = load_state_into_model(kan, args.kan_model)
        baseline = None
        if args.baseline_model:
            baseline = SimpleCNN(num_classes=10, use_kan_head=False)
            baseline = load_state_into_model(baseline, args.baseline_model)

        kan_col = {}
        base_col = {}

        kan_hooks = register_hooks_for_splines(kan, kan_col)
        base_hooks = []
        if baseline is not None:
            base_hooks = register_hooks_for_mlp_postact(baseline, base_col)

        collect_on_grid(kan, "cnn", testloader)
        if baseline is not None:
            collect_on_grid(baseline, "cnn", testloader)

        for h in kan_hooks: h.remove()
        for h in base_hooks: h.remove()

        # For plotting we will use sample index as x-axis
        sample_idx = np.arange(sum(1 for _ in testloader) * 64)
        save_activation_heatmaps(kan_col, sample_idx, os.path.join(save_dir, "kan_cnn"), prefix="kan", max_units=args.max_units)
        if baseline is not None:
            save_activation_heatmaps(base_col, sample_idx, os.path.join(save_dir, "baseline_cnn"), prefix="baseline", max_units=args.max_units)

        print("CIFAR activation response saved under:", save_dir)

if __name__ == "__main__":
    main()
