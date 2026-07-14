import os
import json
import torch
import numpy as np

from .registry import get_extractor
from .fid import precompute_real_stats, compute_fid, compute_class_fid
from .generation import generate_balanced_samples, generate_class_samples, save_samples
from .precision_recall import compute_precision_recall


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
    imbalanced_class_idx=None,
    majority_class_idx=None,
    image_size=100,
    ode_method="dopri5",
    ode_steps=50,
):
    if feature_extractors is None:
        feature_extractors = ["openphenom"]
    if isinstance(feature_extractors, str):
        feature_extractors = [feature_extractors]

    model_fn = ema.forward_with_cfg if cfg_scale > 1.0 else ema.forward

    if majority_class_idx is None and imbalanced_class_idx is not None:
        majority_class_idx = (imbalanced_class_idx + 1) % num_classes

    print(f"Generating {samples_per_class} samples per class for {num_classes} classes...")
    gen_images, gen_labels = generate_balanced_samples(
        model_fn, transport_sampler, num_classes, null_idx,
        samples_per_class, cfg_scale, device, image_size,
        ode_method, ode_steps,
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

        if imbalanced_class_idx is not None and real_stats["per_class"] is not None:
            imb_mu, imb_sigma, _ = real_stats["per_class"][imbalanced_class_idx]
            class_mask = gen_labels == imbalanced_class_idx
            if class_mask.any():
                class_feats = gen_features_all[class_mask]
                imb_fid = compute_class_fid(class_feats, imb_mu, imb_sigma)
                results[f"{ext_name}/fid/imbalanced"] = imb_fid
                results[f"{ext_name}/cmd/imbalanced"] = _class_mean_distance(class_feats, imb_mu)
                results[f"{ext_name}/diversity/imbalanced"] = _intra_class_diversity(class_feats)

        if majority_class_idx is not None and real_stats["per_class"] is not None:
            maj_mu, maj_sigma, _ = real_stats["per_class"][majority_class_idx]
            class_mask = gen_labels == majority_class_idx
            if class_mask.any():
                class_feats = gen_features_all[class_mask]
                maj_fid = compute_class_fid(class_feats, maj_mu, maj_sigma)
                results[f"{ext_name}/fid/majority"] = maj_fid
                results[f"{ext_name}/cmd/majority"] = _class_mean_distance(class_feats, maj_mu)
                results[f"{ext_name}/diversity/majority"] = _intra_class_diversity(class_feats)

        real_feats = real_stats.get("global_feats")
        if real_feats is not None:
            precision, recall = compute_precision_recall(real_feats, feats_np)
            results[f"{ext_name}/precision"] = precision
            results[f"{ext_name}/recall"] = recall

        print(f"  {ext_name}/fid/balanced = {balanced_fid:.4f}")

    results_path = os.path.join(experiment_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {results_path}")

    return results
