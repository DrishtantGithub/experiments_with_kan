# src/tabular/plot_tabular.py
import matplotlib.pyplot as plt
import numpy as np
import os

def save_tabular_plots(y_test, preds_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # preds_dict = {"MLP": y_pred, "KAN": y_pred_kan, ...}

    for model_name, preds in preds_dict.items():
        # Scatter plot
        plt.figure(figsize=(6,5))
        plt.scatter(y_test, preds, s=12, alpha=0.6)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title(f"{model_name} — Predicted vs Actual")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{model_name}_scatter.png"))
        plt.close()

        # Error histogram
        err = preds - y_test
        plt.figure(figsize=(6,5))
        plt.hist(err, bins=30, alpha=0.7)
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.title(f"{model_name} — Error Distribution")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{model_name}_error_hist.png"))
        plt.close()
