# src/analysis/efficiency_summary.py
"""
Efficiency summary for models in the project.

Outputs:
  - results/efficiency/efficiency_summary.csv
  - results/efficiency/efficiency_table.tex
  - results/efficiency/efficiency_barplot.png

What it computes:
  - params (count)
  - forward_time_ms (average over multiple forwards)
  - model_file_size_kb (if model .pth exists)
  - macs (if ptflops installed) or None

It also tries to pick up existing CSV results for tabular/NLP if available
(e.g. results/tabular_*/... or results/nlp_imdb/imdb_metrics.csv) and appends
human-friendly notes from them.

Usage:
  python -m src.analysis.efficiency_summary --device cpu --repeat 200

Note: CPU timings vary by machine. Use --device cuda if you want GPU timings.
"""
import argparse
import os
import time
import io
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# import your models
from src.models.mlp import SimpleMLP
from src.models.cnn_with_kan import SimpleCNN

# optional ptflops
try:
    from ptflops import get_model_complexity_info
    HAS_PTFLOPS = True
except Exception:
    HAS_PTFLOPS = False

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def measure_forward_time(model, sample_input, device='cpu', repeat=200, warmup=20):
    model.to(device)
    model.eval()
    sample_input = sample_input.to(device)
    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)
    # measure
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(sample_input)
    t1 = time.perf_counter()
    avg = (t1 - t0) / repeat
    return avg * 1000.0  # ms

def model_file_size(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return p.stat().st_size / 1024.0  # KB

def try_macs(model, input_res=(1,3,32)):
    if not HAS_PTFLOPS:
        return None
    # ptflops expects model on cpu and takes shape as (C,H,W)
    try:
        macs, params = get_model_complexity_info(model, input_res, as_strings=False,
                                                print_per_layer_stat=False, verbose=False)
        return macs
    except Exception:
        return None

def make_cifar_sample(batch_size=1):
    # CIFAR shaped input: (B,3,32,32)
    x = torch.randn(batch_size, 3, 32, 32, dtype=torch.float32)
    return x

def make_toy_sample(batch_size=1):
    # toy MLP input: (B,1)
    x = torch.randn(batch_size, 1, dtype=torch.float32)
    return x

def safe_load_state_dict(model, pth_path):
    p = Path(pth_path)
    if not p.exists():
        return model
    state = torch.load(pth_path, map_location="cpu")
    if isinstance(state, dict) and ('state_dict' in state or 'model' in state):
        if 'state_dict' in state and isinstance(state['state_dict'], dict):
            model.load_state_dict(state['state_dict'], strict=False)
        elif 'model' in state and isinstance(state['model'], dict):
            model.load_state_dict(state['model'], strict=False)
        else:
            # try direct
            try:
                model.load_state_dict(state, strict=False)
            except Exception:
                pass
    else:
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            pass
    return model

def collect_existing_metrics(results_dir):
    """
    Try to collect some existing metrics to add to 'notes' column.
    Looks for:
     - results/tabular_*/... CSVs
     - results/nlp_imdb/imdb_metrics.csv
    """
    notes = []
    # NLP IMDB
    imdb = Path("results/nlp_imdb/imdb_metrics.csv")
    if imdb.exists():
        try:
            df = pd.read_csv(imdb)
            notes.append(f"IMDB sample: {len(df)} rows")
        except Exception:
            notes.append("IMDB metrics found")
    # Tabular (housing/energy)
    tab_dirs = list(Path("results").glob("tabular_*"))
    for t in tab_dirs:
        # try known filenames
        for name in ("housing_metrics.csv", "energy_metrics.csv", "tabular_metrics.csv"):
            p = t / name
            if p.exists():
                notes.append(f"{t.name}: {name}")
    return "; ".join(notes)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--repeat", type=int, default=200)
    p.add_argument("--save-dir", type=str, default="./results/efficiency")
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    repeat = args.repeat

    rows = []

    # Toy MLP baseline
    m_toy_mlp = SimpleMLP(input_dim=1, hidden_sizes=[64,64,32], num_classes=1, activation='relu')
    pth_toy_mlp = Path("results/toy_relu/toy_model.pth")
    m_toy_mlp = safe_load_state_dict(m_toy_mlp, pth_toy_mlp)
    params = count_params(m_toy_mlp)
    sample = make_toy_sample(batch_size=8)
    fw = measure_forward_time(m_toy_mlp, sample, device=device, repeat=repeat)
    size_kb = model_file_size(pth_toy_mlp)
    rows.append({
        "model": "Toy MLP",
        "params": params,
        "macs": try_macs(m_toy_mlp, input_res=(1,)),
        "forward_time_ms": fw,
        "size_kb": size_kb,
        "notes": ""
    })

    # Toy KAN
    m_toy_kan = SimpleMLP(input_dim=1, hidden_sizes=[64,64,32], num_classes=1, activation='kan',
                          kan_params={'n_knots':21,'x_min':-3.0,'x_max':3.0})
    pth_toy_kan = Path("results/toy_kan/toy_model.pth")
    m_toy_kan = safe_load_state_dict(m_toy_kan, pth_toy_kan)
    params = count_params(m_toy_kan)
    fw = measure_forward_time(m_toy_kan, sample, device=device, repeat=repeat)
    size_kb = model_file_size(pth_toy_kan)
    rows.append({
        "model": "Toy KAN (21 knots)",
        "params": params,
        "macs": try_macs(m_toy_kan, input_res=(1,)),
        "forward_time_ms": fw,
        "size_kb": size_kb,
        "notes": ""
    })

    # CIFAR CNN baseline
    m_cnn = SimpleCNN(num_classes=10, use_kan_head=False)
    pth_cnn = Path("results/cifar_relu/cifar_model.pth")
    m_cnn = safe_load_state_dict(m_cnn, pth_cnn)
    params = count_params(m_cnn)
    sample = make_cifar_sample(batch_size=8)
    fw = measure_forward_time(m_cnn, sample, device=device, repeat=repeat)
    size_kb = model_file_size(pth_cnn)
    rows.append({
        "model": "CIFAR CNN (baseline)",
        "params": params,
        "macs": try_macs(m_cnn, input_res=(3,32,32)),
        "forward_time_ms": fw,
        "size_kb": size_kb,
        "notes": ""
    })

    # CIFAR CNN + KAN head
    m_cnn_kan = SimpleCNN(num_classes=10, use_kan_head=True, kan_params={'n_knots':21,'x_min':-5.0,'x_max':5.0})
    pth_cnn_kan = Path("results/cifar_kan/cifar_model.pth")
    m_cnn_kan = safe_load_state_dict(m_cnn_kan, pth_cnn_kan)
    params = count_params(m_cnn_kan)
    fw = measure_forward_time(m_cnn_kan, sample, device=device, repeat=repeat)
    size_kb = model_file_size(pth_cnn_kan)
    rows.append({
        "model": "CIFAR CNN + KAN head",
        "params": params,
        "macs": try_macs(m_cnn_kan, input_res=(3,32,32)),
        "forward_time_ms": fw,
        "size_kb": size_kb,
        "notes": ""
    })

    # Add knot ablation variants if models saved
    for nkn in (11,41):
        m = SimpleMLP(input_dim=1, hidden_sizes=[64,64,32], num_classes=1, activation='kan',
                      kan_params={'n_knots':nkn,'x_min':-3.0,'x_max':3.0})
        pth = Path(f"results/ablation/knots/toy_kn{nkn}_model.pth")
        m = safe_load_state_dict(m, pth)
        params = count_params(m)
        fw = measure_forward_time(m, make_toy_sample(batch_size=8), device=device, repeat=repeat)
        size_kb = model_file_size(pth)
        rows.append({
            "model": f"Toy KAN ({nkn} knots)",
            "params": params,
            "macs": try_macs(m, input_res=(1,)),
            "forward_time_ms": fw,
            "size_kb": size_kb,
            "notes": "ablation"
        })

    # Attempt to include tabular / nlp notes if CSVs exist
    notes = collect_existing_metrics("results")
    if notes:
        rows.append({
            "model": "Other pipelines (notes)",
            "params": None,
            "macs": None,
            "forward_time_ms": None,
            "size_kb": None,
            "notes": notes
        })

    df = pd.DataFrame(rows)
    csv_path = save_dir / "efficiency_summary.csv"
    df.to_csv(csv_path, index=False)
    print("Saved CSV ->", csv_path)

    # LaTeX table
    tex_lines = []
    tex_lines.append("\\begin{tabular}{lrrrrl}")
    tex_lines.append("\\toprule")
    tex_lines.append("Model & Params & MACs & Forward (ms) & Size (KB) & Notes \\\\")
    tex_lines.append("\\midrule")
    for _, r in df.iterrows():
        params = int(r['params']) if not (pd.isna(r['params']) or r['params'] is None) else "--"
        macs = f"{r['macs']:.2e}" if (r['macs'] is not None and not pd.isna(r['macs'])) else "--"
        fms = f"{r['forward_time_ms']:.3f}" if (r['forward_time_ms'] is not None and not pd.isna(r['forward_time_ms'])) else "--"
        size = f"{r['size_kb']:.1f}" if (r['size_kb'] is not None and not pd.isna(r['size_kb'])) else "--"

        notes = r['notes'] if not pd.isna(r['notes']) else ""
        tex_lines.append(f"{r['model']} & {params} & {macs} & {fms} & {size} & {notes} \\\\")
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")

    tex_path = save_dir / "efficiency_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines))
    print("Saved LaTeX table ->", tex_path)

    # barplot of params
    try:
        import matplotlib.pyplot as plt
        df_plot = df.dropna(subset=['params']).copy()
        if len(df_plot) > 0:
            plt.figure(figsize=(8,4))
            plt.bar(df_plot['model'], df_plot['params'])
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Params")
            plt.tight_layout()
            out = save_dir / "efficiency_barplot_params.png"
            plt.savefig(out, dpi=200)
            plt.close()
            print("Saved barplot ->", out)
    except Exception:
        pass

    print("Efficiency summary complete. CSV + tex + plots are in", save_dir)

if __name__ == "__main__":
    main()
