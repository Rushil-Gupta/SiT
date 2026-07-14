"""
Compute evaluation embeddings for any registered feature extractor.

Generates ops_embeddings_{extractor}.npy files that can be loaded
directly by metrics/fid.py for fast FID evaluation.

Usage:
    # Single GPU
    python compute_eval_embeddings.py --data-dir ./dataset/funk --extractor cell_dino

    # Multi-GPU (all available)
    python compute_eval_embeddings.py --data-dir ./dataset/funk --extractor dinov2 --num-gpus all

    # Specific GPU count
    python compute_eval_embeddings.py --data-dir ./dataset/funk --extractor inception --num-gpus 4

    # List available extractors
    python compute_eval_embeddings.py --list-extractors

Output:
    {data-dir}/ops_embeddings_{extractor}.npy   — (N, D) float32, row-aligned with ops_dataset.npz
    {data-dir}/ops_eval_meta_{extractor}.json   — metadata (embed_dim, total_samples, etc.)
"""

import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm
import torch.multiprocessing as mp

from metrics.registry import get_extractor, list_extractors


def _encode_partition(extractor_name, file_paths, batch_size, device, output_path):
    """Encode a partition of images and save to disk. Returns (n_samples, embed_dim)."""
    from dataset.ops_dataset import GlobalMinMaxNorm

    extractor = get_extractor(extractor_name, device=device)
    # normalize = GlobalMinMaxNorm()

    n_samples = len(file_paths)
    embed_dim = extractor.dim
    all_embeddings = np.zeros((n_samples, embed_dim), dtype=np.float32)

    idx = 0
    for i in tqdm(range(0, n_samples, batch_size),
                  desc=f"[{device}] Encoding", leave=False):
        batch_paths = file_paths[i : i + batch_size]
        batch = []
        for path in batch_paths:
            image = np.load(path).astype(np.float32)
            image = torch.from_numpy(image)
            # image = normalize(image)  # GlobalMinMaxNorm → [0, 1]
            batch.append(image)
        batch_tensor = torch.stack(batch).to(device)
        # encode() calls _preprocess() internally 
        feats = extractor.encode(batch_tensor, gen=False)

        all_embeddings[idx : idx + len(batch)] = feats.cpu().numpy()
        idx += len(batch)

    np.save(output_path, all_embeddings)
    print(f"[{device}] Saved {n_samples:,} embeddings ({embed_dim}d) to {output_path}")
    return n_samples, embed_dim


def _worker_fn(rank, args, results):
    """Worker process for multi-GPU encoding."""
    num_gpus = args.num_gpus
    device = f"cuda:{rank}"

    data_dir = args.data_dir
    npz_path = os.path.join(data_dir, "ops_dataset.npz")
    data = np.load(npz_path, allow_pickle=True)
    total_samples = len(data["file_paths"])
    file_paths = list(data["file_paths"])

    chunk_size = (total_samples + num_gpus - 1) // num_gpus
    start = rank * chunk_size
    end = min((rank + 1) * chunk_size, total_samples)

    print(f"[{device}] Processing indices {start:,} to {end:,} ({end - start:,} samples)")

    chunk_paths = file_paths[start:end]
    temp_path = os.path.join(data_dir, f"ops_embeddings_{args.extractor}_gpu{rank}.npy")
    n_samples, embed_dim = _encode_partition(
        args.extractor, chunk_paths, args.batch_size, device, temp_path
    )

    results[rank] = {
        "output_path": temp_path,
        "start": start,
        "end": end,
        "n_samples": n_samples,
        "embed_dim": embed_dim,
    }


def run_multi_gpu(args):
    """Encode across multiple GPUs using mp.spawn."""
    num_gpus = args.num_gpus
    data_dir = args.data_dir

    npz_path = os.path.join(data_dir, "ops_dataset.npz")
    data = np.load(npz_path, allow_pickle=True)
    total_samples = len(data["file_paths"])

    print(f"Running multi-GPU encoding on {num_gpus} GPUs...")

    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(
        _worker_fn,
        args=(args, results),
        nprocs=num_gpus,
        join=True,
    )

    # Combine partial embeddings
    print("\nCombining embeddings from all GPUs...")
    embed_dim = results[0]["embed_dim"]
    all_embeddings = np.zeros((total_samples, embed_dim), dtype=np.float32)

    for rank in range(num_gpus):
        r = results[rank]
        partial = np.load(r["output_path"])
        all_embeddings[r["start"] : r["end"]] = partial
        os.remove(r["output_path"])

    return all_embeddings


def run_single_gpu(args):
    """Encode on a single GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = args.data_dir

    npz_path = os.path.join(data_dir, "ops_dataset.npz")
    data = np.load(npz_path, allow_pickle=True)
    total_samples = len(data["file_paths"])
    file_paths = list(data["file_paths"])

    print(f"Encoding {total_samples:,} samples on {device}...")

    _, embed_dim = _encode_partition(
        args.extractor, file_paths, args.batch_size, device,
        os.path.join(data_dir, f"ops_embeddings_{args.extractor}.npy"),
    )

    # Load back and return (single-GPU saves directly)
    return np.load(os.path.join(data_dir, f"ops_embeddings_{args.extractor}.npy"))


def main(args):
    if args.list_extractors:
        print("Available extractors:")
        for name in list_extractors():
            print(f"  {name}")
        return

    data_dir = args.data_dir
    extractor_name = args.extractor

    # Check for existing file
    emb_path = os.path.join(data_dir, f"ops_embeddings_{extractor_name}.npy")
    if os.path.exists(emb_path) and not args.force:
        print(f"Embeddings already exist: {emb_path}")
        print("Use --force to recompute.")
        return

    # Determine GPU count
    num_available = torch.cuda.device_count()
    if args.num_gpus == "all":
        num_gpus = num_available
    else:
        num_gpus = int(args.num_gpus)

    args.num_gpus = num_gpus

    # Run encoding
    if num_gpus > 1:
        all_embeddings = run_multi_gpu(args)
    else:
        all_embeddings = run_single_gpu(args)

    # Save final embeddings (multi-GPU path returns array, single-GPU already saved)
    if num_gpus > 1:
        np.save(emb_path, all_embeddings)
        print(f"Saved combined embeddings: {emb_path} ({all_embeddings.shape})")

    # Save metadata
    # npz_path = os.path.join(data_dir, "ops_dataset.npz")
    # data = np.load(npz_path, allow_pickle=True)
    meta = {
        "extractor": extractor_name,
        "embed_dim": int(all_embeddings.shape[1]),
        "total_samples": int(len(all_embeddings)),
        "num_gpus": num_gpus,
        "batch_size": args.batch_size,
    }
    meta_path = os.path.join(data_dir, f"ops_eval_meta_{extractor_name}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute evaluation embeddings for any registered feature extractor."
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing ops_dataset.npz")
    parser.add_argument("--extractor", type=str, default=None,
                        help="Extractor name (e.g. mae_minmax, cell_dino, dinov2, inception)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for encoding")
    parser.add_argument("--num-gpus", type=str, default="all",
                        help="Number of GPUs (default: 1, or 'all')")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if embeddings already exist")
    parser.add_argument("--list-extractors", action="store_true",
                        help="List available extractors and exit")
    args = parser.parse_args()

    if not args.list_extractors and args.extractor is None:
        parser.error("--extractor is required (or use --list-extractors)")

    main(args)
