"""
Standalone evaluation script for SiT models.
Generates samples and computes metrics, or re-evaluates saved samples.

Usage:
    # Full generate + evaluate
    python eval_fid.py --ckpt path/to/checkpoint.pt --model SiT-XL/2 \
        --data-path ./dataset --dataset funk --image-size 100 \
        --embedding-suffix _minmax \
        --feature-extractor mae_minmax \
        --samples-per-class 50 --out-dir ./eval_results

    # Re-evaluate saved samples with different extractor
    python eval_fid.py --load-samples ./eval_results/eval_samples \
        --feature-extractor cell_dino \
        --out-dir ./eval_results/cell_dino
"""
import torch
import argparse
import json
import os
import sys
import numpy as np
from copy import deepcopy

from models import SiT_models, LabelEmbedder
from download import find_model
from transport import create_transport, Sampler
from dataset import OPSDataset
from metrics.registry import get_extractor, list_extractors
from metrics.fid import precompute_real_stats, compute_fid, compute_class_fid
from metrics.generation import load_samples
from metrics.precision_recall import compute_precision_recall
from train_utils import parse_transport_args, parse_ode_args


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    # --- Load or use saved samples ---
    if args.load_samples:
        print(f"Loading saved samples from {args.load_samples}")
        gen_images, gen_labels, pert_map, saved_selected_indices = load_samples(args.load_samples)
        num_classes = len(gen_labels.unique())
        print(f"Loaded {len(gen_images)} images across {num_classes} classes")
        dataset = None
        data_dir = os.path.join(args.data_path, args.dataset)
        # Use saved selected_original_indices if available, else fall back to CLI arg
        if saved_selected_indices is not None:
            selected_indices_for_stats = saved_selected_indices
        elif args.max_perturbations is not None:
            # Reconstruct from dataset
            dataset = OPSDataset(data_dir, max_perturbations=args.max_perturbations, seed=0)
            selected_indices_for_stats = dataset.selected_original_indices
        else:
            selected_indices_for_stats = None
    else:
        # Load model and dataset
        assert args.ckpt is not None, "Must provide --ckpt or --load-samples"
        data_dir = os.path.join(args.data_path, args.dataset)
        dataset = OPSDataset(data_dir, max_perturbations=args.max_perturbations,
                             seed=0)
        num_classes = dataset.get_num_perturbations()
        perturbation_map = dataset.get_perturbation_map()
        selected_indices_for_stats = dataset.selected_original_indices

        print(f"Loading model checkpoint from {args.ckpt}...")
        model = SiT_models[args.model](
            input_size=args.image_size,
            num_classes=num_classes,
            use_frozen_embed=args.use_frozen_embed,
            embed_dim=args.embed_dim,
            cond_dim=args.cond_dim,
            in_channels=4,
        ).to(device)
        model = torch.compile(model)
        state_dict = find_model(args.ckpt)
        model.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict)
        ema = deepcopy(model).to(device)
        if "ema" in state_dict:
            ema.load_state_dict(state_dict["ema"])
        ema.eval()

        # Load class means if needed
        if not isinstance(model.y_embedder, LabelEmbedder):
            suffix = args.embedding_suffix
            means_path = os.path.join(data_dir, f"ops_class_means{suffix}.npy")
            if os.path.isfile(means_path):
                full_means = np.load(means_path)
                if args.use_frozen_embed:
                    K = num_classes
                    embeddings = np.zeros((K + 2, args.cond_dim), dtype=np.float32)
                    embeddings[:K] = full_means[:-1]
                    embeddings[K] = full_means[-1]
                    model.y_embedder.embeddings.copy_(torch.from_numpy(embeddings))
                else:
                    model.y_embedder.class_means.copy_(torch.from_numpy(full_means))
                print(f"Loaded class means from {means_path}")

        transport = create_transport(args.path_type, args.prediction,
                                     args.loss_weight, args.train_eps, args.sample_eps)
        transport_sampler = Sampler(transport)

        null_idx = model.null_idx
        model_fn = ema.forward_with_cfg if args.cfg_scale > 1.0 else ema.forward

        print(f"Generating {args.samples_per_class} samples per class...")
        gen_images, gen_labels = [], []
        from metrics.generation import generate_balanced_samples
        gen_images, gen_labels = generate_balanced_samples(
            model_fn, transport_sampler, num_classes, null_idx,
            args.samples_per_class, args.cfg_scale, device, args.image_size,
        )
        print(f"Generated {len(gen_images)} total images")

    # --- Evaluate with extractor(s) ---
    out_dir = args.out_dir or args.load_samples or "eval_results"
    os.makedirs(out_dir, exist_ok=True)

    extractors = list_extractors() if args.feature_extractor == "all" else [args.feature_extractor]
    all_results = {}

    for ext_name in extractors:
        print(f"\nComputing metrics with: {ext_name}")
        extractor = get_extractor(ext_name, device=device)

        # Extract features from generated images
        gen_feats = []
        for i in range(0, len(gen_images), 64):
            batch = gen_images[i:i+64].to(device)
            feats = extractor.encode(batch)
            gen_feats.append(feats.cpu())
        gen_feats = torch.cat(gen_feats, dim=0)

        # Compute real stats
        real_stats = precompute_real_stats(
            extractor, data_dir, num_classes,
            embedding_suffix=args.embedding_suffix,
            samples_per_class=4500,
            cache_dir=data_dir,
            selected_original_indices=selected_indices_for_stats,
        )

        feats_np = gen_feats.numpy()
        gen_mu = feats_np.mean(axis=0)
        gen_centered = feats_np - gen_mu
        gen_sigma = (gen_centered.T @ gen_centered) / max(len(feats_np) - 1, 1)

        balanced_fid = compute_fid(
            gen_mu, gen_sigma,
            real_stats["global_mu"], real_stats["global_sigma"]
        )
        all_results[f"{ext_name}/fid/balanced"] = balanced_fid

        # Per-class if available
        if real_stats["per_class"] is not None:
            for c_name, c_idx in [("imbalanced", args.imbalanced_class),
                                    ("majority", args.majority_class)]:
                if c_idx is not None and c_idx < len(real_stats["per_class"]):
                    c_mu, c_sigma, _ = real_stats["per_class"][c_idx]
                    class_mask = gen_labels == c_idx
                    if class_mask.any():
                        c_feats = gen_feats[class_mask]
                        c_fid = compute_class_fid(c_feats, c_mu, c_sigma)
                        all_results[f"{ext_name}/fid/{c_name}"] = c_fid
                        c_mean = c_feats.mean(dim=0).numpy()
                        all_results[f"{ext_name}/cmd/{c_name}"] = float(
                            np.linalg.norm(c_mean - c_mu)
                        )
                        norms = c_feats / c_feats.norm(dim=1, keepdim=True).clip(min=1e-8)
                        cos_sim = norms @ norms.T
                        triu = torch.triu(cos_sim, diagonal=1)
                        n_pairs = max((len(c_feats) * (len(c_feats) - 1)) // 2, 1)
                        all_results[f"{ext_name}/diversity/{c_name}"] = float(
                            1.0 - triu.sum() / n_pairs
                        )

        if real_stats.get("global_feats") is not None:
            precision, recall = compute_precision_recall(real_stats["global_feats"],
                                                           feats_np)
        else:
            precision, recall = None, None
            # Precision/recall needs actual features, not just mu/sigma
            # Skip if we don't have full real features cached
        print(f"  {ext_name}/fid/balanced = {balanced_fid:.4f}")

    # Save results
    results_path = os.path.join(out_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--load-samples", type=str, default=None,
                        help="Path to saved eval_samples directory (skip sampling)")
    parser.add_argument("--model", type=str, default="SiT-XL/2",
                        choices=list(SiT_models.keys()))
    parser.add_argument("--image-size", type=int, default=100)
    parser.add_argument("--data-path", type=str, default="./dataset",
                        help="Root dataset directory")
    parser.add_argument("--dataset", type=str, default="funk")
    parser.add_argument("--embedding-suffix", type=str, default="")
    parser.add_argument("--feature-extractor", type=str, default="mae_minmax",
                        choices=list_extractors() + ["all"])
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--use-frozen-embed", action="store_true", default=False)
    parser.add_argument("--cond-dim", type=int, default=384)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-perturbations", type=int, default=None,
                        help="Subset to this many perturbations (must match training)")
    parser.add_argument("--imbalanced-class", type=int, default=None)
    parser.add_argument("--majority-class", type=int, default=None)
    parse_transport_args(parser)
    parse_ode_args(parser)
    args = parser.parse_args()
    main(args)
