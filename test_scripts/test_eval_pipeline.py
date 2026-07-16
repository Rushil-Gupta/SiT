"""
End-to-end check for metrics.evaluate.evaluate_model: generates a small
number of samples from a trained checkpoint, computes FID/precision-recall/
MMD/KID with every registered extractor, and asserts the results are
populated and finite. Exercises the VAE-decode path (metrics/generation.py)
for --use-latent checkpoints in addition to the ordinary pixel-space path.

Keep --eval-samples-per-class tiny -- this is a wiring check, not a sample
quality eval. It needs at least 5 total generated/real samples per extractor
for precision/recall's k=5 nearest-neighbor search to work.

Usage:
    python test_scripts/test_eval_pipeline.py \\
        --ckpt /path/to/checkpoint.pt --model SiT-S/2 \\
        --ops-data-dir /path/to/dataset --dataset funk \\
        --eval-samples-per-class 4

    # latent-space checkpoint:
    python test_scripts/test_eval_pipeline.py \\
        --ckpt /path/to/latent_checkpoint.pt --model SiT-S/2 \\
        --ops-data-dir /path/to/dataset --dataset funk \\
        --use-latent --latent-dir /path/to/latent_dir \\
        --vae-checkpoint /path/to/finetuned_vae.pt \\
        --eval-samples-per-class 4
"""
import argparse
import json
import math
import os
import tempfile
from copy import deepcopy

import numpy as np
import torch

from models import SiT_models, LabelEmbedder
from download import find_model
from transport import create_transport, Sampler
from dataset import OPSDataset, OPSLatentDataset
from vae import FunkVAE
from metrics.registry import list_extractors
from metrics.evaluate import evaluate_model
from train_utils import parse_transport_args


def test_eval_pipeline(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    data_dir = os.path.join(args.ops_data_dir, args.dataset)

    vae = None
    scale_factor = None
    pixel_image_size = None
    if args.use_latent:
        assert args.latent_dir is not None and args.vae_checkpoint is not None, \
            "--latent-dir and --vae-checkpoint are required with --use-latent"
        dataset = OPSLatentDataset(
            data_dir, args.latent_dir, max_perturbations=args.max_perturbations, seed=0,
        )
        in_channels, model_input_size, _ = dataset.latent_shape
        vae = FunkVAE(checkpoint_path=args.vae_checkpoint, device=device)
        vae.eval()
        scale_factor = dataset.scale_factor
        pixel_image_size = args.image_size
    else:
        dataset = OPSDataset(
            data_dir, max_perturbations=args.max_perturbations, seed=0,
        )
        in_channels = 4
        model_input_size = args.image_size

    num_classes = dataset.get_num_perturbations()

    print(f"Loading checkpoint from {args.ckpt}...")
    model = SiT_models[args.model](
        input_size=model_input_size,
        num_classes=num_classes,
        use_frozen_embed=args.use_frozen_embed,
        cond_dim=args.cond_dim,
        in_channels=in_channels,
    ).to(device)
    model = torch.compile(model)
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict)
    ema = deepcopy(model).to(device)
    if "ema" in state_dict:
        ema.load_state_dict(state_dict["ema"])
    ema.eval()

    if not isinstance(model.y_embedder, LabelEmbedder):
        means_path = os.path.join(data_dir, f"ops_class_means{args.embedding_suffix}.npy")
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

    transport = create_transport(args.path_type, args.prediction, args.loss_weight,
                                  args.train_eps, args.sample_eps)
    transport_sampler = Sampler(transport)

    extractors = list_extractors()
    assert len(extractors) > 0, "No extractors registered -- metrics/extractors import is broken"
    print(f"Evaluating with extractors: {extractors}")

    with tempfile.TemporaryDirectory() as experiment_dir:
        results = evaluate_model(
            ema=ema,
            transport_sampler=transport_sampler,
            dataset=dataset,
            data_dir=data_dir,
            num_classes=num_classes,
            null_idx=model.null_idx,
            device=device,
            experiment_dir=experiment_dir,
            feature_extractors=extractors,
            samples_per_class=args.eval_samples_per_class,
            cfg_scale=args.cfg_scale,
            embedding_suffix=args.embedding_suffix,
            image_size=model_input_size,
            vae=vae,
            scale_factor=scale_factor,
            pixel_image_size=pixel_image_size,
        )
        assert os.path.exists(os.path.join(experiment_dir, "eval_results.json")), \
            "evaluate_model did not write eval_results.json"

    print(json.dumps(results, indent=2))

    expected_suffixes = ["fid/balanced", "precision", "recall", "mmd", "kid_mean", "kid_std"]
    for ext_name in extractors:
        for suffix in expected_suffixes:
            key = f"{ext_name}/{suffix}"
            assert key in results, f"Missing expected result key: {key}"
            value = results[key]
            assert value is not None and math.isfinite(value), \
                f"{key} is not a finite value: {value}"

    print("\n=== eval pipeline test passed! ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", type=str, default="SiT-S/2")
    parser.add_argument("--ops-data-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="funk")
    parser.add_argument("--image-size", type=int, default=100)
    parser.add_argument("--embedding-suffix", type=str, default="")
    parser.add_argument("--use-frozen-embed", action="store_true", default=False)
    parser.add_argument("--cond-dim", type=int, default=384)
    parser.add_argument("--max-perturbations", type=int, default=None)
    parser.add_argument("--eval-samples-per-class", type=int, default=4,
                        help="Keep tiny -- this is a wiring check, not a quality eval")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--use-latent", action="store_true", default=False)
    parser.add_argument("--latent-dir", type=str, default=None)
    parser.add_argument("--vae-checkpoint", type=str, default=None)
    parse_transport_args(parser)
    args = parser.parse_args()

    test_eval_pipeline(args)


if __name__ == "__main__":
    main()
