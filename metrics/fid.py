import os
import pickle
import numpy as np
import torch
from scipy import linalg as scipy_linalg


def compute_fid(mu1, sigma1, mu2, sigma2):
    mu1 = np.asarray(mu1, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    diff = mu1 - mu2

    covmean, _ = scipy_linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def _filter_and_remap(embeddings, perturbation_indices, selected_original_indices):
    """Filter embeddings to selected perturbations and remap labels from global to compact indices.

    Args:
        embeddings: (N, D) array of embeddings for all samples.
        perturbation_indices: (N,) array of global perturbation indices (0..1450).
        selected_original_indices: list/array of global indices that were selected
            (from OPSDataset.selected_original_indices).

    Returns:
        filtered_embeddings: (M, D) array with only selected samples.
        remapped_labels: (M,) array with compact labels in 0..len(selected)-1.
    """
    selected = np.asarray(selected_original_indices)
    keep = np.isin(perturbation_indices, selected)
    filtered_embeddings = embeddings[keep]
    old_to_new = {int(old): new for new, old in enumerate(selected)}
    remapped_labels = np.array([old_to_new[int(idx)] for idx in perturbation_indices[keep]],
                               dtype=np.int64)
    return filtered_embeddings, remapped_labels


def _per_class_stats(embeddings, labels, num_classes, samples_per_class,
                     pr_samples_per_class=200):
    per_class = []
    pr_feats = []
    for c in range(num_classes):
        mask = labels == c
        feats = embeddings[mask]
        if len(feats) == 0:
            continue
        if len(feats) > samples_per_class:
            rng = np.random.RandomState(0)
            idx = rng.choice(len(feats), samples_per_class, replace=False)
            feats = feats[idx]
        mu = feats.mean(axis=0)
        centered = feats - mu
        sigma = (centered.T @ centered) / max(len(feats) - 1, 1)
        per_class.append((mu, sigma, len(feats)))
        if len(feats) > pr_samples_per_class:
            rng = np.random.RandomState(0)
            idx = rng.choice(len(feats), pr_samples_per_class, replace=False)
            pr_feats.append(feats[idx])
        else:
            pr_feats.append(feats)
    pr_feats = np.concatenate(pr_feats, axis=0)
    global_mu = pr_feats.mean(axis=0)
    global_centered = pr_feats - global_mu
    global_sigma = (global_centered.T @ global_centered) / max(len(pr_feats) - 1, 1)
    return per_class, global_mu, global_sigma, pr_feats


def _load_embeddings(data_dir, extractor_name, embedding_suffix=""):
    """Load pre-computed embeddings for any extractor.

    Tries new naming convention first (ops_embeddings_{name}.npy),
    then falls back to legacy MAE naming (ops_embeddings{suffix}.npy).
    Raises FileNotFoundError with helpful message if neither exists.
    """
    # New naming: ops_embeddings_{extractor_name}.npy
    emb_path = os.path.join(data_dir, f"ops_embeddings_{extractor_name}.npy")
    if os.path.exists(emb_path):
        return np.load(emb_path, mmap_mode="r")

    # Legacy fallback: ops_embeddings{suffix}.npy (MAE only)
    if embedding_suffix:
        legacy_path = os.path.join(data_dir, f"ops_embeddings{embedding_suffix}.npy")
        if os.path.exists(legacy_path):
            return np.load(legacy_path, mmap_mode="r")

    raise FileNotFoundError(
        f"No pre-computed embeddings found for extractor '{extractor_name}'.\n"
        f"Expected: {emb_path}\n"
        f"Run:\n"
        f"  python compute_eval_embeddings.py --data-dir {data_dir} "
        f"--extractor {extractor_name}"
    )


def precompute_real_stats(extractor, data_dir, num_classes, embedding_suffix="",
                          samples_per_class=500, cache_dir=None, force=False,
                          selected_original_indices=None):
    """Compute real distribution statistics for FID evaluation.

    Loads pre-computed embeddings (from compute_eval_embeddings.py) and
    computes per-class and global statistics.

    Args:
        extractor: FeatureExtractor instance.
        data_dir: Directory containing ops_dataset.npz and embeddings.
        num_classes: Number of classes in the (possibly subsetted) dataset.
        embedding_suffix: Legacy suffix for MAE embeddings (e.g. "_minmax").
        samples_per_class: Max samples per class for computing stats.
        cache_dir: Directory to cache results (defaults to data_dir).
        force: If True, recompute even if cache exists.
        selected_original_indices: If provided, filter global perturbation indices
            to this subset and remap labels to compact 0..len-1. Obtained from
            OPSDataset.selected_original_indices.
    """
    if cache_dir is None:
        cache_dir = data_dir

    # Build cache key: include subset size to avoid stale caches
    subset_tag = ""
    if selected_original_indices is not None and len(selected_original_indices) < 1451:
        subset_tag = f"_n{len(selected_original_indices)}"
    cache_path = os.path.join(
        cache_dir,
        f"real_stats_{extractor.name}{embedding_suffix}{subset_tag}_spc{samples_per_class}.pkl"
    )

    if os.path.exists(cache_path) and not force:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Load pre-computed embeddings (same path for all extractors now)
    embeddings = _load_embeddings(data_dir, extractor.name, embedding_suffix)

    # Load global labels
    npz_path = os.path.join(data_dir, "ops_dataset.npz")
    data = np.load(npz_path, allow_pickle=True)
    perturbation_indices = data["perturbation_indices"]

    # Filter and remap if subsetting was used
    if selected_original_indices is not None:
        embeddings, perturbation_indices = _filter_and_remap(
            embeddings, perturbation_indices, selected_original_indices
        )

    per_class, global_mu, global_sigma, pr_feats = _per_class_stats(
        embeddings, perturbation_indices, num_classes, samples_per_class
    )

    result = {
        "per_class": per_class,
        "global_mu": global_mu,
        "global_sigma": global_sigma,
        "global_feats": pr_feats,
        "samples_per_class": samples_per_class,
        "num_classes": num_classes,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


def compute_class_fid(gen_features, real_mu, real_sigma):
    gen_features = gen_features.cpu().numpy() if torch.is_tensor(gen_features) else gen_features
    gen_mu = gen_features.mean(axis=0)
    gen_centered = gen_features - gen_mu
    gen_sigma = (gen_centered.T @ gen_centered) / max(len(gen_features) - 1, 1)
    return compute_fid(gen_mu, gen_sigma, real_mu, real_sigma)
