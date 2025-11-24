# src/utils/utils.py
import torch, time
import os
from pathlib import Path

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, save_dir, name='checkpoint.pth'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    torch.save(state, path)
    return path

def timeit(func, *args, **kwargs):
    t0 = time.time()
    out = func(*args, **kwargs)
    return out, time.time() - t0
