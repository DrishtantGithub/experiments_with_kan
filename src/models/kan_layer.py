# src/models/kan_layer.py
"""
Spline-based activation for KAN-like layers.

Implementation notes:
- Supports per-feature (channel) learnable knot y-values.
- Uses uniform, fixed x-knot positions between x_min and x_max.
- Forward does vectorized linear interpolation.
- Designed for fully-connected MLP-like usage. For CNN channels, use features=channels.
"""
import torch
import torch.nn as nn

class SplineActivation(nn.Module):
    def __init__(self, n_knots=21, x_min=-5.0, x_max=5.0, per_channel=True, in_features=None):
        super().__init__()
        assert n_knots >= 2
        self.n_knots = n_knots
        self.x_min = x_min
        self.x_max = x_max
        self.register_buffer('x_knots', torch.linspace(x_min, x_max, n_knots))  # (K,)
        self.dx = (x_max - x_min) / (n_knots - 1)

        if per_channel:
            assert in_features is not None, "Provide in_features for per_channel True"
            # y: (in_features, n_knots)
            self.y = nn.Parameter(torch.zeros(in_features, n_knots))
            # initialize to identity-ish mapping
            xs = torch.linspace(x_min, x_max, n_knots)
            self.y.data = xs.unsqueeze(0).repeat(in_features, 1)
            self.per_channel = True
            self.in_features = in_features
        else:
            # y: (n_knots,)
            init = torch.linspace(x_min, x_max, n_knots)
            self.y = nn.Parameter(init.clone())
            self.per_channel = False

    def forward(self, x):
        # Keep shape, but operate on last dimension as features
        orig_shape = x.shape
        # collapse leading dims to 2D: (N, F)
        if x.dim() == 1:
            x_flat = x.unsqueeze(0)
        else:
            n_feat = orig_shape[-1]
            x_flat = x.reshape(-1, n_feat)  # (N, F)

        x_clamped = x_flat.clamp(self.x_min, self.x_max)
        pos = (x_clamped - self.x_min) / self.dx
        idx = pos.floor().long().clamp(0, self.n_knots - 2)  # (N, F)
        t = (pos - idx.float()).clamp(0.0, 1.0)

        if self.per_channel:
            # y: (F, K) -> expand to (N, F, K)
            y_exp = self.y.unsqueeze(0).expand(x_flat.shape[0], -1, -1)  # (N, F, K)
            idx_exp = idx.unsqueeze(-1)  # (N, F, 1)
            y_l = torch.gather(y_exp, 2, idx_exp).squeeze(-1)  # (N, F)
            y_r = torch.gather(y_exp, 2, idx_exp + 1).squeeze(-1)
        else:
            # y: (K,) -> y[idx], y[idx+1]
            y_l = self.y[idx]
            y_r = self.y[idx + 1]

        out_flat = (1.0 - t) * y_l + t * y_r
        out = out_flat.view(orig_shape)
        return out
