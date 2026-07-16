"""MMD (Maximum Mean Discrepancy) and KID (Kernel Inception Distance) — kernel
two-sample tests over the same embeddings already used for FID/precision-recall.
Both operate on whatever feature extractor is active (openphenom/inception/
dinov2/cell_dino); unlike FID they make no Gaussian assumption on the feature
distribution, and KID in particular is more reliable at small sample counts.
"""
import numpy as np
import torch


def _to_tensor(x, device="cpu"):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float64)
    return torch.as_tensor(np.asarray(x), dtype=torch.float64, device=device)


def _rbf_kernel(x, y, sigma=None):
    """Gaussian RBF kernel matrix between rows of x (n,d) and y (m,d).
    If sigma is not given, uses the median pairwise-distance heuristic on x."""
    x_sq = (x * x).sum(dim=1, keepdim=True)
    y_sq = (y * y).sum(dim=1, keepdim=True)
    dist_sq = (x_sq + y_sq.T - 2.0 * (x @ y.T)).clamp(min=0.0)

    if sigma is None:
        with torch.no_grad():
            n = x.shape[0]
            idx = torch.randperm(n)[:min(n, 500)]
            sub = x[idx]
            sub_sq = (sub * sub).sum(dim=1, keepdim=True)
            sub_dist_sq = (sub_sq + sub_sq.T - 2.0 * (sub @ sub.T)).clamp(min=0.0)
            nonzero = sub_dist_sq[sub_dist_sq > 0]
            median_sq = nonzero.median() if nonzero.numel() > 0 else torch.tensor(1.0, dtype=x.dtype)
            sigma = torch.sqrt(median_sq / 2.0).clamp(min=1e-8)

    return torch.exp(-dist_sq / (2.0 * sigma ** 2))


def _polynomial_kernel(x, y, degree=3, gamma=None, coef0=1.0):
    """Standard KID kernel (Binkowski et al. 2018): gamma defaults to 1/dim."""
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * (x @ y.T) + coef0) ** degree


def _unbiased_mmd2(kxx, kyy, kxy):
    """Unbiased MMD^2 U-statistic estimator (excludes diagonal self-similarity terms)."""
    n = kxx.shape[0]
    m = kyy.shape[0]
    sum_kxx = (kxx.sum() - kxx.diagonal().sum()) / max(n * (n - 1), 1)
    sum_kyy = (kyy.sum() - kyy.diagonal().sum()) / max(m * (m - 1), 1)
    sum_kxy = kxy.sum() / max(n * m, 1)
    return float((sum_kxx + sum_kyy - 2.0 * sum_kxy).item())


def compute_mmd(real_feats, gen_feats, kernel="rbf", sigma=None,
                degree=3, gamma=None, coef0=1.0, device="cpu"):
    """Unbiased MMD^2 between real and generated feature sets."""
    x = _to_tensor(real_feats, device)
    y = _to_tensor(gen_feats, device)
    if kernel == "rbf":
        kxx = _rbf_kernel(x, x, sigma)
        kyy = _rbf_kernel(y, y, sigma)
        kxy = _rbf_kernel(x, y, sigma)
    elif kernel == "polynomial":
        kxx = _polynomial_kernel(x, x, degree, gamma, coef0)
        kyy = _polynomial_kernel(y, y, degree, gamma, coef0)
        kxy = _polynomial_kernel(x, y, degree, gamma, coef0)
    else:
        raise ValueError(f"Unknown kernel '{kernel}', expected 'rbf' or 'polynomial'.")
    return _unbiased_mmd2(kxx, kyy, kxy)


def compute_kid(real_feats, gen_feats, degree=3, gamma=None, coef0=1.0,
                subset_size=1000, n_subsets=100, device="cpu", seed=0):
    """Kernel Inception Distance (Binkowski et al. 2018): polynomial-kernel
    unbiased MMD^2, averaged over random subsets. Returns (mean, std) across
    subsets — the conventional way KID is reported — which avoids the cost of
    a full pairwise kernel matrix at large sample counts.
    """
    x_full = _to_tensor(real_feats, device)
    y_full = _to_tensor(gen_feats, device)
    subset_size = min(subset_size, x_full.shape[0], y_full.shape[0])
    rng = np.random.RandomState(seed)

    scores = []
    for _ in range(n_subsets):
        x_idx = rng.choice(x_full.shape[0], subset_size, replace=False)
        y_idx = rng.choice(y_full.shape[0], subset_size, replace=False)
        x = x_full[x_idx]
        y = y_full[y_idx]
        kxx = _polynomial_kernel(x, x, degree, gamma, coef0)
        kyy = _polynomial_kernel(y, y, degree, gamma, coef0)
        kxy = _polynomial_kernel(x, y, degree, gamma, coef0)
        scores.append(_unbiased_mmd2(kxx, kyy, kxy))

    scores = np.array(scores)
    return float(scores.mean()), float(scores.std())
