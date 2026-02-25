import torch
from torch import nn
from models.CNN import norm_embeddings


def compute_distances4comps(embeddings, comps, metric="euclidean"):
    distances = {}
    for comp in comps:
        e1, e2 = norm_embeddings(embeddings[comp[0]]), norm_embeddings(embeddings[comp[1]])
        distance = compute_distance(e1, e2, metric)
        distances[str(comp)] = distance
    return distances

def compute_distance(e1, e2, metric):
    if metric == "euclidean":
        distance = nn.functional.pairwise_distance(e1, e2).item()
    elif metric == "cosine":
        distance = nn.functional.cosine_similarity(e1, e2).item()
    else:
        raise ValueError(f"Unknown metric {metric}")
    return distance

def normalize_distances(distances):
    """
    Convert distances to similarity in [0, 1]
    """
    distances_list = torch.tensor(list(distances.values()), dtype=torch.float32)
    max_dist = distances_list.max()
    similarities = 1 - distances_list / max_dist
    return similarities.tolist()