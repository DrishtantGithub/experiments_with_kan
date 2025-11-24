# src/train/train_toy.py
"""
Train script for toy regression to validate KAN spline activation.
Run: python src/train/train_toy.py --quick
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.toy_regression import ToySinDataset
from src.models.mlp import SimpleMLP
from src.utils.seed import set_seed
from src.utils.utils import get_device, save_checkpoint
import os
import matplotlib.pyplot as plt

def train(args):
    set_seed(args.seed)
    device = get_device()
    ds = ToySinDataset(n_samples=2000, noise=0.08, kind='sine', seed=args.seed)
    if args.quick:
        ds = ToySinDataset(n_samples=500, noise=0.08, kind='sine', seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = SimpleMLP(input_dim=1, hidden_sizes=[64,64,32], num_classes=1,
                      activation=('kan' if args.activation=='kan' else args.activation),
                      kan_params={'n_knots':21, 'x_min':-3.0, 'x_max':3.0})
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    losses=[]
    for epoch in range(args.epochs):
        model.train()
        batch_loss = 0.0
        for x,y in dl:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_loss += loss.item()*x.size(0)
        avg = batch_loss / len(dl.dataset)
        losses.append(avg)
        if epoch % args.log_every == 0:
            print(f"Epoch {epoch}/{args.epochs} loss={avg:.6f}")
    # save model
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'toy_model.pth'))

    # plot
    xs = torch.linspace(-3,3,400).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        ys = model(xs).cpu().numpy()
    plt.figure(figsize=(6,4))
    plt.plot(xs.cpu().numpy(), ys, label='model')
    # also plot dataset points
    import numpy as np
    X = ds.x
    Y = ds.y
    plt.scatter(X, Y, s=6, alpha=0.6, label='data')
    plt.legend()
    plt.title('Toy regression fit')
    plt.savefig(os.path.join(save_dir, 'toy_fit.png'))
    print("Saved results in", save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='run quick debug')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--activation', type=str, default='kan', choices=['relu','gelu','kan'])
    parser.add_argument('--save_dir', type=str, default='./results/toy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=10)
    args = parser.parse_args()
    if args.quick:
        args.epochs = 50
    train(args)
