# src/train/train_cifar.py
"""
Quick CIFAR-10 training script for baseline (CNN) and CNN+KAN head.
Usage:
  python src/train/train_cifar.py --quick --use_kan
"""
import argparse, os
import torch, torch.nn as nn
from src.data.cifar10_loader import get_cifar10_dataloaders
from src.models.cnn_with_kan import SimpleCNN
from src.utils.seed import set_seed
from src.utils.utils import get_device, save_checkpoint
from tqdm import tqdm

def train(args):
    set_seed(args.seed)
    device = get_device()
    trainloader, testloader = get_cifar10_dataloaders(batch_size=args.batch_size, quick=args.quick)
    model = SimpleCNN(num_classes=10, use_kan_head=args.use_kan, kan_params={'n_knots':21, 'x_min':-5.0, 'x_max':5.0})
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running=0.0
        for xb, yb in tqdm(trainloader, desc=f"Train E{epoch}"):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()*xb.size(0)
        print(f"Epoch {epoch} train_loss {running/len(trainloader.dataset):.4f}")

        # quick eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in testloader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        acc = correct / total
        print(f"Epoch {epoch} val_acc {acc:.4f}")

    # save
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'cifar_model.pth'))
    print("Saved model to", save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--use_kan', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./results/cifar')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.quick:
        args.epochs = 3
        args.batch_size = 64
    train(args)
