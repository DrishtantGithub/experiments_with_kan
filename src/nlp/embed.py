# src/nlp/embed.py
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

def load_embedder():
    # small, fast, CPU-friendly model
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(embedder, texts, batch_size=32):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vecs = embedder.encode(batch, convert_to_numpy=True)
        all_vecs.append(vecs)

    # FIRST stack using numpy
    arr = np.vstack(all_vecs)

    # THEN convert to torch tensor
    return torch.tensor(arr, dtype=torch.float32)

