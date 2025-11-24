# src/analysis/add_residual_to_efficiency.py

import torch
import pandas as pd
from pathlib import Path

from src.train.simple_cnn_residual import SimpleCNNResidual
from src.analysis.efficiency_summary import (
    measure_forward_time, count_params, try_macs, model_file_size
)

def main():
    device = "cpu"
    repeat = 60  # fast and safe
    save_dir = Path("results/efficiency")

    # Read existing summary
    csv_path = save_dir / "efficiency_summary.csv"
    df = pd.read_csv(csv_path)

    print("Loaded existing efficiency_summary.csv")
    
    # Create residual KAN model
    model = SimpleCNNResidual(
        num_classes=10,
        head_type="residual_kan",
        kan_params={'n_knots':21,'x_min':-5.0,'x_max':5.0}
    )
    
    pth = Path("results/cifar_residual/residual_kan/cifar_model.pth")
    if pth.exists():
        state = torch.load(pth, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("Loaded residual KAN model weights.")
    else:
        print("WARNING: residual KAN .pth not found — measuring untrained model.")

    # Measure params, size, forward time, macs
    params = count_params(model)
    sample = torch.randn(8, 3, 32, 32)

    print("Measuring forward pass time...")
    fw = measure_forward_time(model, sample, device=device, repeat=repeat)

    print("Measuring model size...")
    size_kb = model_file_size(pth)

    print("Calculating MACs (if ptflops available)...")
    macs = try_macs(model, (3, 32, 32))

    # Add to DataFrame
    new_row = {
        "model": "CIFAR Residual KAN Head",
        "params": params,
        "macs": macs,
        "forward_time_ms": fw,
        "size_kb": size_kb,
        "notes": "extension"
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save updated CSV
    df.to_csv(csv_path, index=False)
    print("\nUpdated efficiency summary saved to:")
    print(csv_path)

    print("\nLast 5 rows:")
    print(df.tail())

if __name__ == "__main__":
    main()
