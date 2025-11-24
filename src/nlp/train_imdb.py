# src/nlp/train_imdb.py
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from src.nlp.load_imdb import IMDBDataset
from src.nlp.embed import load_embedder, embed_texts
from src.models.mlp import SimpleMLP
from src.utils.seed import set_seed
from src.utils.utils import save_checkpoint
from src.nlp.plot_confusion import save_confusion_matrix


def train_classifier(X_train, y_train, X_test, y_test, activation="relu"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = torch.utils.data.TensorDataset(
        X_train, y_train.unsqueeze(1)
    )
    dl = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = SimpleMLP(
        input_dim=X_train.shape[1],
        hidden_sizes=[128, 64],
        num_classes=1,
        activation=activation,
        kan_params={"n_knots": 21, "x_min": -3, "x_max": 3}
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(10):  # quick training
        model.train()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device).float()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(device))
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_test.numpy(), preds)
    f1 = f1_score(y_test.numpy(), preds)

    return model, preds, acc, f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--save_dir", type=str, default="./results/nlp_imdb")
    args = p.parse_args()

    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    max_train = 200 if args.quick else 20000
    max_test = 200 if args.quick else 5000

    # 1. Load IMDB texts
    train_ds = IMDBDataset("train", max_samples=max_train)
    test_ds = IMDBDataset("test", max_samples=max_test)

    train_texts = [t for t, _ in train_ds]
    test_texts = [t for t, _ in test_ds]
    train_labels = torch.tensor([l for _, l in train_ds])
    test_labels = torch.tensor([l for _, l in test_ds])

    # 2. Embedding step
    embedder = load_embedder()
    X_train = embed_texts(embedder, train_texts)
    X_test = embed_texts(embedder, test_texts)

    # 3. Train baseline MLP
    mlp_model, mlp_preds, mlp_acc, mlp_f1 = train_classifier(
        X_train, train_labels, X_test, test_labels, activation="relu"
    )

    # 4. Train KAN classifier
    kan_model, kan_preds, kan_acc, kan_f1 = train_classifier(
        X_train, train_labels, X_test, test_labels, activation="kan"
    )

    # 5. Save metrics CSV
    csv_path = os.path.join(args.save_dir, "imdb_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("model,accuracy,f1\n")
        f.write(f"MLP,{mlp_acc},{mlp_f1}\n")
        f.write(f"KAN,{kan_acc},{kan_f1}\n")
    print("Saved metrics:", csv_path)

    # 6. Save confusion matrices
    save_confusion_matrix(test_labels.numpy(), mlp_preds, os.path.join(args.save_dir, "mlp_confusion.png"))
    save_confusion_matrix(test_labels.numpy(), kan_preds, os.path.join(args.save_dir, "kan_confusion.png"))

    print("All IMDB results saved in:", args.save_dir)


if __name__ == "__main__":
    main()
