# src/nlp/plot_confusion.py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def save_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center", fontsize=12)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
