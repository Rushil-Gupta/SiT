import os
import json
from collections import defaultdict

import torch
import numpy as np

from .registry import get_extractor
from .fid import precompute_real_stats, compute_fid, compute_class_fid
from .generation import generate_balanced_samples, generate_class_samples, save_samples
from .precision_recall import compute_precision_recall
from .mmd_kid import compute_mmd, compute_kid


def _intra_class_diversity(features):
    features = features.cpu().numpy() if torch.is_tensor(features) else features
    n = len(features)
    if n < 2:
        return 0.0
    norms = features / np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-8)
    cos_sim = norms @ norms.T
    triu = np.triu(cos_sim, k=1)
    return float(1.0 - triu.sum() / max((n * (n - 1)) // 2, 1))


def _class_mean_distance(gen_features, real_mu):
    gen_features = gen_features.cpu().numpy() if torch.is_tensor(gen_features) else gen_features
    gen_mu = gen_features.mean(axis=0)
    return float(np.linalg.norm(gen_mu - real_mu))


def evaluate_model(
    ema,
    transport_sampler,
    dataset,
    data_dir,
    num_classes,
    null_idx,
    device,
    experiment_dir,
    feature_extractors=None,
    samples_per_class=50,
    cfg_scale=1.0,
    embedding_suffix="",
    image_size=100,
    ode_method="dopri5",
    ode_steps=50,
    vae=None,
    scale_factor=None,
    pixel_image_size=None,
):
    """
    image_size: spatial size of the noise tensor fed to the model — the
        latent size (e.g. 16) when `vae` is given (--use-latent models),
        otherwise the pixel image size.
    vae, scale_factor, pixel_image_size: pass a loaded FunkVAE (+ the
        dataset's scale_factor + the target pixel resolution, e.g. 100) for
        latent-space models, so generated samples are decoded back to real
        pixel-space images before being saved/passed to any extractor.
        Leave as None for pixel-space models.

    Per-class metrics are grouped by `dataset.class_imbalance_factors` (see
    OPSDataset) into tiers -- e.g. factor 1.0 (no imbalance), 0.1, 0.05, 0.01
    -- rather than hardcoding a single "imbalanced" and "majority" class.
    Each tier's FID/CMD/diversity is the mean (and std, across that tier's
    classes) of the same per-class computation, reusing
    `real_stats["per_class"]` (already computed for every class) and
    `compute_class_fid` (already class-agnostic) -- no new FID math needed.
    """
    if vae is not None:
        assert scale_factor is not None and pixel_image_size is not None, \
            "scale_factor and pixel_image_size are required when vae is provided"
    if feature_extractors is None:
        feature_extractors = ["openphenom"]
    if isinstance(feature_extractors, str):
        feature_extractors = [feature_extractors]

    model_fn = ema.forward_with_cfg if cfg_scale > 1.0 else ema.forward

    class_factors = getattr(dataset, "class_imbalance_factors", None)
    tiers = defaultdict(list)
    if class_factors is not None:
        for class_idx, factor in enumerate(class_factors):
            tiers[round(float(factor), 6)].append(class_idx)

    print(f"Generating {samples_per_class} samples per class for {num_classes} classes...")
    gen_images, gen_labels = generate_balanced_samples(
        model_fn, transport_sampler, num_classes, null_idx,
        samples_per_class, cfg_scale, device, image_size,
        ode_method, ode_steps,
        vae=vae, scale_factor=scale_factor, pixel_image_size=pixel_image_size,
    )

    sample_dir = os.path.join(experiment_dir, "eval_samples")
    pert_map = dataset.get_perturbation_map() if hasattr(dataset, "get_perturbation_map") else None
    selected_indices = getattr(dataset, 'selected_original_indices', None)
    save_samples(gen_images, gen_labels, sample_dir, perturbation_map=pert_map,
                 selected_original_indices=selected_indices)
    print(f"Saved generated samples to {sample_dir}")

    results = {}
    for ext_name in feature_extractors:
        print(f"Computing metrics with extractor: {ext_name}")
        extractor = get_extractor(ext_name, device=device)

        real_stats = precompute_real_stats(
            extractor, data_dir, num_classes,
            embedding_suffix=embedding_suffix,
            samples_per_class=samples_per_class,
            cache_dir=data_dir,
            selected_original_indices=selected_indices,
        )

        gen_features_list = []
        for i in range(0, len(gen_images), 64):
            batch = gen_images[i:i+64].to(device)
            feats = extractor.encode(batch, gen=True)
            gen_features_list.append(feats.cpu())
        gen_features_all = torch.cat(gen_features_list, dim=0)

        feats_np = gen_features_all.numpy()
        gen_mu = feats_np.mean(axis=0)
        gen_centered = feats_np - gen_mu
        gen_sigma = (gen_centered.T @ gen_centered) / max(len(feats_np) - 1, 1)

        balanced_fid = compute_fid(
            gen_mu, gen_sigma,
            real_stats["global_mu"], real_stats["global_sigma"]
        )
        results[f"{ext_name}/fid/balanced"] = balanced_fid

        if real_stats["per_class"] is not None:
            for factor in sorted(tiers.keys(), reverse=True):
                tier_fids, tier_cmds, tier_diversities = [], [], []
                for class_idx in tiers[factor]:
                    class_mu, class_sigma, _ = real_stats["per_class"][class_idx]
                    class_mask = gen_labels == class_idx
                    if not class_mask.any():
                        continue
                    class_feats = gen_features_all[class_mask]
                    tier_fids.append(compute_class_fid(class_feats, class_mu, class_sigma))
                    tier_cmds.append(_class_mean_distance(class_feats, class_mu))
                    tier_diversities.append(_intra_class_diversity(class_feats))

                if not tier_fids:
                    continue
                tier_name = f"{factor:g}"
                results[f"{ext_name}/fid/tier_{tier_name}"] = float(np.mean(tier_fids))
                results[f"{ext_name}/fid/tier_{tier_name}_std"] = float(np.std(tier_fids))
                results[f"{ext_name}/cmd/tier_{tier_name}"] = float(np.mean(tier_cmds))
                results[f"{ext_name}/diversity/tier_{tier_name}"] = float(np.mean(tier_diversities))
                results[f"{ext_name}/tier_{tier_name}_num_classes"] = len(tier_fids)

        real_feats = real_stats.get("global_feats")
        if real_feats is not None:
            precision, recall = compute_precision_recall(real_feats, feats_np)
            results[f"{ext_name}/precision"] = precision
            results[f"{ext_name}/recall"] = recall

            results[f"{ext_name}/mmd"] = compute_mmd(real_feats, feats_np)
            kid_mean, kid_std = compute_kid(real_feats, feats_np)
            results[f"{ext_name}/kid_mean"] = kid_mean
            results[f"{ext_name}/kid_std"] = kid_std

        print(f"  {ext_name}/fid/balanced = {balanced_fid:.4f}")

    results_path = os.path.join(experiment_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {results_path}")

    return results
