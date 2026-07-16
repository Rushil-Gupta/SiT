"""
Confirms OPSLatentDataset's indexing is wired correctly through subsetting/
imbalance filtering: for a few samples, re-encodes the raw image at
dataset.file_paths[idx] directly through FunkVAE and checks it matches (up to
the scale_factor) the latent OPSLatentDataset returns for the same idx --
i.e. that self.sample_indices correctly maps a post-filtering idx back to its
row in the master latents.npy.

Run this with the *same* --vae-checkpoint used to produce the latents in
--latent-dir, since the check re-encodes from scratch and compares.

Usage:
    python test_scripts/test_latent_dataset_indexing.py \\
        --ops-data-dir /scratch/rg5218/SiT --latent-dir /path/to/latent_dir \\
        --vae-checkpoint /path/to/finetuned_vae.pt --max-perturbations 10
"""
import argparse
import json
import os
import tempfile

import numpy as np
import torch

from dataset import OPSDataset, OPSLatentDataset
from dataset.ops_dataset import GlobalMinMaxNorm
from vae import FunkVAE


def to_pm1(x: torch.Tensor) -> torch.Tensor:
    return x * 2 - 1


def test_indexing(data_dir, latent_dir, vae_checkpoint, max_perturbations, imbalance_factor, seed, num_check, device):
    fd, config_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({"tiers": {str(imbalance_factor): 1}}, f)

    try:
        dataset = OPSLatentDataset(
            data_dir, latent_dir,
            max_perturbations=max_perturbations, class_distribution_file=config_path, seed=seed,
        )
    finally:
        os.remove(config_path)
    print(f"  Dataset size after filtering: {len(dataset)}")

    norm = GlobalMinMaxNorm()
    vae = FunkVAE(checkpoint_path=vae_checkpoint, device=device)
    vae.eval()

    rng = np.random.RandomState(1)
    check_indices = rng.choice(len(dataset), size=min(num_check, len(dataset)), replace=False)

    max_abs_diff = 0.0
    with torch.no_grad():
        for idx in check_indices:
            cached_latent, _ = dataset[int(idx)]

            path = dataset.file_paths[idx]
            image = np.load(path).astype(np.float32)
            x = torch.from_numpy(image).unsqueeze(0).to(device)
            x = to_pm1(norm(x))
            x_padded = vae.pad(x)
            z, _ = vae.encode(x_padded, sample=False)
            fresh_latent = (z[0].cpu() * dataset.scale_factor)

            diff = (cached_latent - fresh_latent).abs().max().item()
            max_abs_diff = max(max_abs_diff, diff)
            print(f"  idx={idx}: max abs diff vs freshly-encoded latent = {diff:.6f}")

    print(f"  Max abs diff across {len(check_indices)} samples: {max_abs_diff:.6f}")
    assert max_abs_diff < 1e-3, \
        "Cached latent doesn't match a fresh re-encode -- sample_indices lookup may be misaligned " \
        "(or --vae-checkpoint doesn't match the one used for precompute_latents.py)"
    print("  [PASS] Cached latents match fresh re-encodes -- indexing survives filtering.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops-data-dir", required=True)
    parser.add_argument("--dataset", default="funk")
    parser.add_argument("--latent-dir", required=True)
    parser.add_argument("--vae-checkpoint", required=True,
                         help="Must be the same checkpoint used by precompute_latents.py")
    parser.add_argument("--max-perturbations", type=int, default=10,
                         help="Exercise the subsetting path (default: small subset)")
    parser.add_argument("--imbalance-factor", type=float, default=0.3,
                         help="Exercise the imbalance-filtering path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-check", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = os.path.join(args.ops_data_dir, args.dataset)

    print("=== Latent dataset indexing test ===")
    test_indexing(
        data_dir, args.latent_dir, args.vae_checkpoint,
        args.max_perturbations, args.imbalance_factor, args.seed, args.num_check, device,
    )


if __name__ == "__main__":
    main()
