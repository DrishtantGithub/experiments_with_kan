# src/analysis/plot_splines.py
"""
Plot learned spline activations from saved models.

Usage (from project root):
  # Toy model (regression)
  python -m src.analysis.plot_splines --model ./results/toy_kan/toy_model.pth --model-type toy

  # CIFAR CNN + KAN head
  python -m src.analysis.plot_splines --model ./results/cifar_kan/cifar_model.pth --model-type cnn

Options:
  --save-dir: directory to save plots (default: ./results/interpretability)
  --max-plots: max number of splines to plot from a layer (default: 32)
"""
import argparse, os
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# import models (module style)
from src.models.mlp import SimpleMLP
from src.models.cnn_with_kan import SimpleCNN
from src.models.kan_layer import SplineActivation

def find_spline_modules(model):
    """Return list of (module_name, module) pairs for each SplineActivation found."""
    out = []
    for name, m in model.named_modules():
        if isinstance(m, SplineActivation):
            out.append((name, m))
    return out

def plot_spline_module(name, spline: SplineActivation, save_dir, max_plots=32):
    """
    Plot the spline curves for a SplineActivation module.
    If per_channel, spline.y has shape (features, n_knots). Otherwise (n_knots,).
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    x_knots = spline.x_knots.detach().cpu().numpy()
    n_knots = spline.n_knots
    per_channel = getattr(spline, 'per_channel', False)
    if per_channel:
        y = spline.y.detach().cpu().numpy()  # (features, n_knots)
        n_features = y.shape[0]
    else:
        y = spline.y.detach().cpu().numpy()[None, :]  # (1, n_knots)
        n_features = 1

    # how many curves to plot
    n_plot = min(n_features, max_plots)
    cols = 4
    rows = int(np.ceil(n_plot / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = np.array(axes).reshape(-1)
    for i in range(n_plot):
        ax = axes[i]
        yi = y[i]
        # linear interpolate for smooth curve for visual clarity
        xs = np.linspace(x_knots.min(), x_knots.max(), 200)
        # numpy interp
        ys = np.interp(xs, x_knots, yi)
        ax.plot(xs, ys, lw=2)
        ax.scatter(x_knots, yi, s=12, alpha=0.8)
        ax.set_title(f"{name} | unit {i}")
        ax.grid(True)
    # hide remaining axes
    for j in range(n_plot, len(axes)):
        axes[j].axis('off')
    plt.suptitle(f"Spline activations - {name} (showing {n_plot}/{n_features})", fontsize=14)
    out_file = os.path.join(save_dir, f"spline_{name.replace('.', '_')}.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_file, dpi=200)
    plt.close(fig)

    # also save knot table for the plotted units
    csv_lines = ["unit," + ",".join([f"knot_{i}" for i in range(n_knots)])]
    for i in range(n_plot):
        row = [str(i)] + [f"{float(v):.6f}" for v in y[i]]
        csv_lines.append(",".join(row))
    csv_path = os.path.join(save_dir, f"spline_{name.replace('.', '_')}.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"Saved {out_file} and {csv_path}")

def load_model_state(model, path):
    state = torch.load(path, map_location="cpu")
    # Try both direct state_dict or full dict
    if isinstance(state, dict) and any(k.startswith('module.') or k in model.state_dict() for k in state.keys()):
        try:
            model.load_state_dict(state)
        except Exception:
            # maybe saved as {'model': state_dict}
            if 'model' in state and isinstance(state['model'], dict):
                model.load_state_dict(state['model'])
            elif 'state_dict' in state and isinstance(state['state_dict'], dict):
                model.load_state_dict(state['state_dict'])
            else:
                # last attempt - strict False
                model.load_state_dict(state, strict=False)
    else:
        # if it fails, try loading into .state_dict keys
        try:
            model.load_state_dict(state)
        except Exception:
            print("Warning: couldn't load state strictly; attempting non-strict load")
            model.load_state_dict(state, strict=False)
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True, help='path to saved model .pth')
    p.add_argument('--model-type', type=str, choices=['toy','cnn'], default='toy', help='toy or cnn')
    p.add_argument('--save-dir', type=str, default='./results/interpretability', help='where to save plots')
    p.add_argument('--max-plots', type=int, default=32, help='max number of unit plots per layer')
    args = p.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    if args.model_type == 'toy':
        # toy model: SimpleMLP(input_dim=1, hidden_sizes=[64,64,32], num_classes=1, activation='kan')
        model = SimpleMLP(input_dim=1, hidden_sizes=[64,64,32], num_classes=1, activation='kan',
                          kan_params={'n_knots':21, 'x_min':-3.0, 'x_max':3.0})
    else:
        # CIFAR CNN: SimpleCNN(num_classes=10, use_kan_head=True)
        model = SimpleCNN(num_classes=10, use_kan_head=True, kan_params={'n_knots':21, 'x_min':-5.0, 'x_max':5.0})

    # load weights
    model = load_model_state(model, model_path)
    model.eval()

    splines = find_spline_modules(model)
    if len(splines) == 0:
        print("No SplineActivation modules found in model.")
        return

    print(f"Found {len(splines)} spline modules. Plotting to {args.save_dir}")
    for name, mod in splines:
        plot_spline_module(name, mod, args.save_dir, max_plots=args.max_plots)

if __name__ == "__main__":
    main()
