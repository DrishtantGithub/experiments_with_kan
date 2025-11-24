# src/models/mlp.py
import torch
import torch.nn as nn
from .kan_layer import SplineActivation

class SimpleMLP(nn.Module):
    """
    Basic fully-connected MLP for classification (or regression if num_classes=1).
    Allows switching activation between ReLU and SplineActivation (KAN).
    """
    def __init__(self, input_dim, hidden_sizes=[128,64], num_classes=10, activation='relu', kan_params=None):
        super().__init__()
        layers = []
        prev = input_dim
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev, h))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'kan':
                # per-channel spline: each hidden unit has its own spline
                kp = kan_params or {}
                layers.append(SplineActivation(in_features=h, **kp))
            else:
                raise ValueError("Unknown activation")
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
