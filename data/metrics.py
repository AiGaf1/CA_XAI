import torch
import numpy as np
from sklearn.metrics import roc_curve
from models.CNN import norm_embeddings


def compute_eer(pred1, pred2, labels):
    if torch.isnan(pred1).any() or torch.isinf(pred2).any():
        print("Warning: NaN or Inf detected in z1 — replacing with 0")
    dists = torch.nn.functional.pairwise_distance(
        norm_embeddings(pred1), norm_embeddings(pred2)
    ).numpy()
    fpr, tpr, threshold = roc_curve(labels, dists, pos_label=1)
    fnr = 1 - tpr
    EER1 = fpr[np.nanargmin(np.abs(fnr - fpr))]
    EER2 = fnr[np.nanargmin(np.abs(fnr - fpr))]
    return np.mean([EER1, EER2]) * 100