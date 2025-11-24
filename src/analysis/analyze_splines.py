import argparse, os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.mlp import SimpleMLP
from src.models.cnn_with_kan import SimpleCNN
from src.models.kan_layer import SplineActivation

def compute_spline_metrics(spline: SplineActivation):
    """
    Compute smoothness metrics for a single SplineActivation module.
    Returns: { 'unit': i, 'tv': value, 'l2_smoothness': value }
    """
    x_knots = spline.x_knots.detach().cpu().numpy()
    y = spline.y.detach().cpu().numpy()
    if y.ndim == 1:  # shape (n_knots,)
        y = y[None, :]

    metrics = []
    for i, yi in enumerate(y):
        xs = np.linspace(x_knots.min(), x_knots.max(), 400)
        ys = np.interp(xs, x_knots, yi)

        # derivative (finite differences)
        dy = np.gradient(ys, xs)

        # smoothness metrics
        tv = np.sum(np.abs(np.diff(dy)))          # total variation of derivative
        l2_smooth = np.mean(dy**2)                # average squared slope

        metrics.append((i, tv, l2_smooth))
    return metrics

def plot_derivative_curves(name, spline: SplineActivation, save_dir, max_plots=16):
    """Plot derivative curves to visualize smoothness."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    x_knots = spline.x_knots.detach().cpu().numpy()
    y = spline.y.detach().cpu().numpy()
    if y.ndim == 1:
        y = y[None, :]

    n_units = min(len(y), max_plots)
    rows = int(np.ceil(n_units/4))
    fig, axes = plt.subplots(rows, 4, figsize=(4*4, rows*3))
    axes = np.array(axes).reshape(-1)

    for i in range(n_units):
        yi = y[i]
        xs = np.linspace(x_knots.min(), x_knots.max(), 400)
        ys = np.interp(xs, x_knots, yi)
        dy = np.gradient(ys, xs)

        ax = axes[i]
        ax.plot(xs, dy, lw=2)
        ax.set_title(f"{name} — Derivative Unit {i}")
        ax.grid(True)

    for j in range(n_units, len(axes)):
        axes[j].axis('off')

    out = os.path.join(save_dir, f"derivatives_{name.replace('.', '_')}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print("Saved", out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--model-type", choices=["toy","cnn"], default="toy")
    p.add_argument("--save-dir", type=str, default="./results/interpretability/depth")
    p.add_argument("--max-plots", type=int, default=16)
    args = p.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    if args.model_type == "toy":
        model = SimpleMLP(1, [64,64,32], 1, activation="kan",
            kan_params={"n_knots":21,"x_min":-3,"x_max":3})
    else:
        model = SimpleCNN(num_classes=10, use_kan_head=True)

    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    # Find spline modules
    spline_modules = [(name, m) for name, m in model.named_modules()
                      if isinstance(m, SplineActivation)]
    print("Found spline modules:", len(spline_modules))

    # Analyze each
    summary = ["module,unit,tv,l2"]
    for (name, spline) in spline_modules:
        metrics = compute_spline_metrics(spline)
        for (unit, tv, l2) in metrics:
            summary.append(f"{name},{unit},{tv:.6f},{l2:.6f}")

        # plot derivative curves
        plot_derivative_curves(name, spline, args.save_dir, args.max_plots)

    # save csv
    csv_path = os.path.join(args.save_dir, "spline_smoothness.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(summary))
    print("Saved:", csv_path)

if __name__ == "__main__":
    main()
