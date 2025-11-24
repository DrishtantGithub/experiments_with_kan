# src/data/toy_regression.py
import numpy as np
from torch.utils.data import Dataset

class ToySinDataset(Dataset):
    def __init__(self, n_samples=1000, noise=0.1, kind='sine', seed=0):
        np.random.seed(seed)
        self.x = np.random.uniform(-3.0, 3.0, size=(n_samples,1)).astype(np.float32)
        if kind == 'sine':
            self.y = np.sin(self.x) + noise * np.random.randn(n_samples,1).astype(np.float32)
        elif kind == 'poly':
            self.y = (self.x**3 - 0.5*self.x**2 + 0.3*self.x) + noise * np.random.randn(n_samples,1).astype(np.float32)
        else:
            raise ValueError("Unknown kind")
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
