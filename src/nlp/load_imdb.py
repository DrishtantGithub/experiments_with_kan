# src/nlp/load_imdb.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import random

class IMDBDataset(Dataset):
    def __init__(self, split="train", max_samples=None, seed=42):
        ds = load_dataset("imdb")[split]
        random.seed(seed)
        self.texts = []
        self.labels = []
        idxs = list(range(len(ds)))
        random.shuffle(idxs)

        for i in idxs:
            if max_samples is not None and len(self.texts) >= max_samples:
                break
            self.texts.append(ds[i]["text"])
            self.labels.append(1 if ds[i]["label"] == 1 else 0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
