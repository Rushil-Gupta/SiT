"""
Encodes every OPS image through the finetuned FunkVAE (see vae.py) and caches
the resulting latents to a single packed array, so SiT training never has to
run the VAE encoder inside the training loop and so we don't create one file
per sample.

This dataset is 1.1M+ images (see dataset/funk/compute_data_range.py). Each
latent is only 4x16x16 float32 (~4KB), so one .npy per sample would mean
1.1M+ small files purely from filesystem/inode overhead -- a cost the raw
images pay because each one is large (~80KB, ~90GB total), which doesn't
apply here: the full packed latent array is only ~4.7GB, small enough to
memory-map as a single file.

Latents are encoded in the *master* (unfiltered) file_paths order from
ops_dataset.npz, so row i of latents.npy corresponds to file_paths[i]. Any
downstream subsetting/imbalance filtering (see OPSDataset/OPSLatentDataset)
maps back to this order via sample_indices rather than re-deriving storage.

Also computes an empirical scaling factor (std of latents) from this dataset,
analogous to Stable Diffusion's 0.18215 constant, since that constant was fit
to natural images and has no reason to be right for microscopy latents.

Encoding is embarrassingly parallel (no gradient sync needed, unlike
finetune_vae.py's training loop), so multi-GPU here uses the mp.spawn
partition-and-combine recipe from compute_eval_embeddings.py rather than DDP:
each worker independently encodes a contiguous slice of file_paths, writing
directly into its own disjoint row range of the shared latents.npy memmap,
then partial (sum, sq_sum, count) stats are combined at the end for the
scale factor. No torchrun/process-group setup needed -- just run the script.

Usage:
    # Single GPU
    python dataset/funk/precompute_latents.py \\
        --data-dir dataset/funk --output-dir dataset/funk/latents \\
        --vae-checkpoint vae_checkpoints/vae_final.pt

    # Multi-GPU (all available)
    python dataset/funk/precompute_latents.py \\
        --data-dir dataset/funk --output-dir dataset/funk/latents \\
        --vae-checkpoint vae_checkpoints/vae_final.pt --num-gpus all

    # Specific GPU count
    python dataset/funk/precompute_latents.py \\
        --data-dir dataset/funk --output-dir dataset/funk/latents \\
        --vae-checkpoint vae_checkpoints/vae_final.pt --num-gpus 4
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.multiprocessing as mp

from dataset import GlobalMinMaxNorm
from vae import FunkVAE


def to_pm1(x: torch.Tensor) -> torch.Tensor:
    return x * 2 - 1


def _encode_partition(file_paths, row_start, vae_checkpoint, device, latents_path, batch_size, report_every):
    """Encodes file_paths (a contiguous slice of the master list) into
    latents_path[row_start : row_start + len(file_paths)]. Returns partial
    (sum, sq_sum, count) over the encoded latents for the scale-factor calc."""
    norm = GlobalMinMaxNorm()
    vae = FunkVAE(checkpoint_path=vae_checkpoint, device=device)
    vae.eval()

    latents_mmap = np.lib.format.open_memmap(latents_path, mode="r+")

    partial_sum = 0.0
    partial_sq_sum = 0.0
    partial_count = 0

    with torch.no_grad():
        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i:i + batch_size]
            images = np.stack([np.load(p).astype(np.float32) for p in batch_paths])
            x = torch.from_numpy(images).to(device)
            x = norm(x)
            x = to_pm1(x)
            x = vae.pad(x)

            z, _ = vae.encode(x, sample=False)  # deterministic (posterior mode) for caching

            partial_sum += z.sum().item()
            partial_sq_sum += (z ** 2).sum().item()
            partial_count += z.numel()

            batch_start = row_start + i
            latents_mmap[batch_start:batch_start + len(batch_paths)] = z.cpu().numpy()

            if (i // batch_size) % report_every == 0:
                print(f"  [{device}] [{i}/{len(file_paths)}]")

    latents_mmap.flush()
    return partial_sum, partial_sq_sum, partial_count


def _worker_fn(rank, args, results):
    """Worker process for multi-GPU encoding: handles one contiguous chunk
    of the master file_paths list, indexed by GPU rank."""
    device = f"cuda:{rank}"

    npz_path = os.path.join(args.data_dir, "ops_dataset.npz")
    data = np.load(npz_path, allow_pickle=True)
    file_paths = list(data["file_paths"])
    total_samples = len(file_paths)

    chunk_size = (total_samples + args.num_gpus - 1) // args.num_gpus
    start = rank * chunk_size
    end = min(start + chunk_size, total_samples)
    print(f"[{device}] Encoding indices {start:,}-{end:,} ({end - start:,} samples)")

    latents_path = os.path.join(args.output_dir, "latents.npy")
    partial_sum, partial_sq_sum, partial_count = _encode_partition(
        file_paths[start:end], start, args.vae_checkpoint, device, latents_path,
        args.batch_size, args.report_every,
    )
    results[rank] = (partial_sum, partial_sq_sum, partial_count)


def run_multi_gpu(args):
    print(f"Running multi-GPU latent precomputation on {args.num_gpus} GPUs...")
    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(_worker_fn, args=(args, results), nprocs=args.num_gpus, join=True)

    total_sum = sum(r[0] for r in results.values())
    total_sq_sum = sum(r[1] for r in results.values())
    total_count = sum(r[2] for r in results.values())
    return total_sum, total_sq_sum, total_count


def run_single_gpu(args, file_paths, latents_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding {len(file_paths):,} samples on {device}...")
    return _encode_partition(
        file_paths, 0, args.vae_checkpoint, device, latents_path,
        args.batch_size, args.report_every,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing ops_dataset.npz (raw images)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write latents.npy + latent_meta.json")
    parser.add_argument("--vae-checkpoint", required=True,
                        help="Path to a finetuned VAE checkpoint from finetune_vae.py")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--report-every", type=int, default=1000)
    parser.add_argument("--num-gpus", type=str, default="all",
                        help="Number of GPUs ('all', an integer, or 1 for single-process)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    npz_path = os.path.join(args.data_dir, "ops_dataset.npz")
    data = np.load(npz_path, allow_pickle=True)
    file_paths = list(data["file_paths"])
    num_samples = len(file_paths)

    num_available = torch.cuda.device_count()
    num_gpus = num_available if args.num_gpus == "all" else int(args.num_gpus)
    args.num_gpus = num_gpus

    latent_shape = (FunkVAE.NUM_CHANNELS, FunkVAE.PAD_TARGET // 8, FunkVAE.PAD_TARGET // 8)
    latents_path = os.path.join(args.output_dir, "latents.npy")

    # Preallocate the packed array up front (single writer) so every worker
    # (or the single process, below) can then open it in r+ mode and write
    # into its own disjoint row range concurrently.
    np.lib.format.open_memmap(
        latents_path, mode="w+", dtype=np.float32, shape=(num_samples, *latent_shape)
    )

    if num_gpus > 1:
        total_sum, total_sq_sum, total_count = run_multi_gpu(args)
    else:
        total_sum, total_sq_sum, total_count = run_single_gpu(args, file_paths, latents_path)

    mean = total_sum / total_count
    std = (total_sq_sum / total_count - mean ** 2) ** 0.5
    scale_factor = 1.0 / std

    meta = {
        "scale_factor": scale_factor,
        "latent_mean": mean,
        "latent_std": std,
        "latent_shape": list(latent_shape),
        "vae_checkpoint": args.vae_checkpoint,
        "num_samples": num_samples,
        "num_gpus": num_gpus,
    }
    with open(os.path.join(args.output_dir, "latent_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {num_samples} latents to {latents_path}")
    print(f"Latent shape: {latent_shape}, scale_factor: {scale_factor:.4f}")


if __name__ == "__main__":
    main()
