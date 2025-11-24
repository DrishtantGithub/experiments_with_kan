# src/models/simple_cnn_residual.py
"""
Simple CNN for CIFAR with optional KAN head and Residual KAN head.

This file defines:
 - SplineActivation is imported from your existing src/models/kan_layer.py
 - SimpleCNNResidual: small convnet -> head

Head modes (head_type):
  - "linear"        : regular linear classifier head
  - "kan"           : KAN head (SplineActivation -> Linear)
  - "residual_kan"  : Residual KAN head: Linear(skip) + Linear(Spline(x))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.models.kan_layer import SplineActivation
except Exception:
    # If import fails, try relative import fallback
    from ..models.kan_layer import SplineActivation  # type: ignore

class ResidualKANHead(nn.Module):
    def __init__(self, in_features, num_classes, n_knots=21, x_min=-5.0, x_max=5.0, per_channel=False):
        super().__init__()
        # Spline activation operates element-wise: we create a module that applies SplineActivation
        self.spline = SplineActivation(n_knots=n_knots, x_min=x_min, x_max=x_max, per_channel=per_channel)
        # two small linear layers
        self.fc_spline = nn.Linear(in_features, num_classes)
        # skip linear from raw features
        self.fc_skip = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: (B, in_features)
        s = self.spline(x)        # shape (B, in_features)
        out1 = self.fc_spline(s)  # (B, num_classes)
        out2 = self.fc_skip(x)    # (B, num_classes)
        return out1 + out2

class KANHead(nn.Module):
    def __init__(self, in_features, num_classes, n_knots=21, x_min=-5.0, x_max=5.0, per_channel=False):
        super().__init__()
        self.spline = SplineActivation(n_knots=n_knots, x_min=x_min, x_max=x_max, per_channel=per_channel)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        s = self.spline(x)
        return self.fc(s)

class SimpleCNNResidual(nn.Module):
    def __init__(self, num_classes=10, head_type="linear", kan_params=None):
        """
        head_type: "linear", "kan", or "residual_kan"
        kan_params: dict with keys n_knots, x_min, x_max, per_channel (optional)
        """
        super().__init__()
        # Simple small convnet suitable for CIFAR quick runs
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),  # -> (B,128,1,1)
        )
        self.feature_dim = 128

        self.head_type = head_type
        if kan_params is None:
            kan_params = {'n_knots':21, 'x_min':-5.0, 'x_max':5.0, 'per_channel': False}
        # instantiate appropriate head
        if head_type == "linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif head_type == "kan":
            self.head = KANHead(in_features=self.feature_dim, num_classes=num_classes,
                                 n_knots=kan_params.get('n_knots',21),
                                 x_min=kan_params.get('x_min', -5.0),
                                 x_max=kan_params.get('x_max', 5.0),
                                 per_channel=kan_params.get('per_channel', False))
        elif head_type == "residual_kan":
            self.head = ResidualKANHead(in_features=self.feature_dim, num_classes=num_classes,
                                        n_knots=kan_params.get('n_knots',21),
                                        x_min=kan_params.get('x_min', -5.0),
                                        x_max=kan_params.get('x_max', 5.0),
                                        per_channel=kan_params.get('per_channel', False))
        else:
            raise ValueError("Unknown head_type: "+str(head_type))

    def forward(self, x):
        # x: (B,3,32,32)
        f = self.features(x)          # (B, 128,1,1)
        f = f.view(f.size(0), -1)     # (B, 128)
        out = self.head(f)            # (B, num_classes)
        return out
