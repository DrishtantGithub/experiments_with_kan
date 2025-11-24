# src/models/cnn_with_kan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kan_layer import SplineActivation

class SimpleCNN(nn.Module):
    """
    Small CNN for CIFAR-10 with option to replace final MLP head with KAN activation.
    This is intentionally small for quick runs; scale up channels for longer experiments.
    """
    def __init__(self, num_classes=10, use_kan_head=False, kan_params=None):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.use_kan_head = use_kan_head
        if use_kan_head:
            # flatten to 128 features, then linear to num_classes implemented as
            # linear layer + per-class spline (interpretable)
            self.fc = nn.Linear(128, num_classes)
            # apply KAN as activation on logits (optional) — we will apply per-class splines to logits
            self.kan = SplineActivation(in_features=num_classes, **(kan_params or {}))
        else:
            self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        if self.use_kan_head:
            out = self.kan(logits)
            return out
        return logits
