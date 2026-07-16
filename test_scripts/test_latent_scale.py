"""
Sanity-checks the cached latents produced by dataset/funk/precompute_latents.py:
confirms the empirical scale factor in latent_meta.json actually brings the
latents to roughly unit variance, and that latents.npy's shape matches the
recorded metadata.

Usage:
    python test_scripts/test_latent_scale.py --latent-dir /path/to/latent_dir
"""
import argparse
import json
import os

import numpy as np


def test_latent_scale(latent_dir, sample_size):
    with open(os.path.join(latent_dir, "latent_meta.json"), "r") as f:
        meta = json.load(f)

    latents = np.load(os.path.join(latent_dir, "latents.npy"), mmap_mode="r")
    print(f"  latents.npy shape: {latents.shape}")
    print(f"  latent_meta.json: num_samples={meta['num_samples']}, latent_shape={meta['latent_shape']}")

    assert latents.shape[0] == meta["num_samples"], \
        f"latents.npy has {latents.shape[0]} rows, but metadata says {meta['num_samples']}"
    assert list(latents.shape[1:]) == meta["latent_shape"], \
        f"latents.npy per-sample shape {latents.shape[1:]} != metadata {meta['latent_shape']}"

    # Random subsample (latents.npy can be large; avoid loading it all into memory)
    rng = np.random.RandomState(0)
    idx = rng.choice(latents.shape[0], size=min(sample_size, latents.shape[0]), replace=False)
    idx.sort()  # mmap fancy-indexing is faster with sorted indices
    sample = np.array(latents[idx], dtype=np.float64)

    raw_std = sample.std()
    scaled_std = (sample * meta["scale_factor"]).std()
    print(f"  Raw latent std (subsample of {len(idx)}): {raw_std:.4f}")
    print(f"  Recorded scale_factor: {meta['scale_factor']:.4f} (== 1/latent_std recorded at precompute time: {1.0 / meta['latent_std']:.4f})")
    print(f"  Std after applying scale_factor: {scaled_std:.4f} (should be close to 1.0)")

    assert 0.5 < scaled_std < 2.0, \
        f"Scaled latent std {scaled_std:.4f} is far from 1.0 -- scale_factor may be stale or mismatched"

    print("  [PASS] latents.npy shape matches metadata and scale_factor normalizes std to ~1.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dir", required=True,
                         help="Directory containing latents.npy + latent_meta.json")
    parser.add_argument("--sample-size", type=int, default=5000,
                         help="Number of latents to randomly subsample for the std check")
    args = parser.parse_args()

    print(f"=== Latent scale factor test ({args.latent_dir}) ===")
    test_latent_scale(args.latent_dir, args.sample_size)


if __name__ == "__main__":
    main()
