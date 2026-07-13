import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def compute_precision_recall(real_features, gen_features, k=5):
    real = real_features.cpu().numpy() if torch.is_tensor(real_features) else real_features
    gen = gen_features.cpu().numpy() if torch.is_tensor(gen_features) else gen_features

    nbrs_real = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(real)
    real_distances, _ = nbrs_real.kneighbors(real)
    real_radii = real_distances[:, -1]

    nbrs_gen = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(gen)
    gen_distances, _ = nbrs_gen.kneighbors(gen)
    gen_radii = gen_distances[:, -1]

    precision = 0.0
    for g in gen:
        dists = np.linalg.norm(real - g, axis=1)
        in_manifold = (dists <= real_radii).any()
        if in_manifold:
            precision += 1.0
    precision /= len(gen)

    recall = 0.0
    for r in real:
        dists = np.linalg.norm(gen - r, axis=1)
        in_manifold = (dists <= gen_radii).any()
        if in_manifold:
            recall += 1.0
    recall /= len(real)

    return float(precision), float(recall)
