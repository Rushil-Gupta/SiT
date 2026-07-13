"""
Generate frozen encoder embeddings for all OPS images and compute per-class means.

Usage:
    # Single GPU (default)
    python generate_embeddings.py --data-dir /path/to/ops_data

    # Multi-GPU (uses all available GPUs)
    python generate_embeddings.py --data-dir /path/to/ops_data --num-gpus all

    # Multi-GPU (specific number)
    python generate_embeddings.py --data-dir /path/to/ops_data --num-gpus 4

This script:
1. Loads the frozen OpenPhenom encoder (one copy per GPU for parallel inference)
2. Runs inference on all images in ops_dataset.npz, partitioned across GPUs
3. Saves individual embeddings (ops_embeddings.npy)
4. Saves per-class mean embeddings including NTC (ops_class_means.npy)
"""

import argparse
import json
import os
import sqlite3
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm
import torch.multiprocessing as mp


class Arcsinh(nn.Module):
    def forward(self, x):
        return torch.arcsinh(x)


class MinMaxNormalize(nn.Module):
    _ch_max = torch.Tensor([65535.0, 16628.0, 65212.0, 65535.0])
    _ch_min = torch.Tensor([0.0, 0.0, 0.0, 0.0])
    def forward(self, x):
        ch_range = self._ch_max[:, None, None] - self._ch_min[:, None, None]
        return (x - self._ch_min[:, None, None]) / ch_range


class OPSImageDataset(Dataset):
    """Dataset that loads raw OPS images for encoder inference."""

    def __init__(self, data_dir, indices=None, preprocessing="arcsinh"):
        npz_path = os.path.join(data_dir, 'ops_dataset.npz')
        data = np.load(npz_path, allow_pickle=True)
        self.all_file_paths = data['file_paths']
        self.all_perturbation_indices = data['perturbation_indices']

        if indices is not None:
            self.file_paths = self.all_file_paths[indices]
            self.perturbation_indices = self.all_perturbation_indices[indices]
        else:
            self.file_paths = self.all_file_paths
            self.perturbation_indices = self.all_perturbation_indices

        # Image transforms: resize for encoder, then preprocessing
        if preprocessing == "arcsinh":
            self.transforms = T.Compose([
                T.Resize((256, 256)),
                Arcsinh(),
                T.Normalize(7., 7.)
            ])
        elif preprocessing == "minmax":
            self.transforms = T.Compose([
                T.Resize((256, 256)),
                MinMaxNormalize(),
            ])
        else:
            raise ValueError(f"Unknown preprocessing: {preprocessing}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = int(self.perturbation_indices[idx])

        image = np.load(path).astype(np.float32)
        image = torch.from_numpy(image)
        image = self.transforms(image)

        return image, label


def get_ntc_paths(data_dir):
    """Return list of file paths for nontargeting perturbations."""
    db_path = os.path.join(data_dir, 'ops_files.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT file_path FROM files WHERE perturbation = 'nontargeting'")
    paths = [row[0] for row in c.fetchall()]
    conn.close()
    return paths


def load_encoder(model_name, device):
    """Load frozen OpenPhenom encoder."""
    from OpenPhenom.huggingface_mae import MAEModel

    print(f"[{device}] Loading encoder: {model_name}")
    model = MAEModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.input_norm = nn.Identity()
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    print(f"[{device}] Encoder loaded")
    return model


def run_inference_on_partition(model, dataset, batch_size, device, output_path):
    """Run encoder inference on a dataset partition and save to output_path."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    embed_dim = 384
    n_samples = len(dataset)
    all_embeddings = np.zeros((n_samples, embed_dim), dtype=np.float32)

    idx = 0
    for images, labels in tqdm(loader, desc=f"Encoding on {device}", leave=False):
        images = images.to(device)

        with torch.no_grad():
            embeddings = model.predict(images)

        all_embeddings[idx:idx + len(images)] = embeddings.cpu().numpy()
        idx += len(images)

    np.save(output_path, all_embeddings)
    print(f"[{device}] Saved {n_samples:,} embeddings to {output_path}")
    return n_samples


def compute_ntc_partial(ntc_paths, batch_size, device, model_name, preprocessing="arcsinh"):
    """Compute partial NTC embedding sum and count."""
    embed_dim = 384
    sum_embedding = np.zeros(embed_dim, dtype=np.float64)
    count = 0

    if preprocessing == "arcsinh":
        transforms = T.Compose([
            T.Resize((256, 256)),
            Arcsinh(),
            T.Normalize(7., 7.)
        ])
    elif preprocessing == "minmax":
        transforms = T.Compose([
            T.Resize((256, 256)),
            MinMaxNormalize(),
        ])
    else:
        raise ValueError(f"Unknown preprocessing: {preprocessing}")

    model = load_encoder(model_name, device)

    for i in range(0, len(ntc_paths), batch_size):
        batch_paths = ntc_paths[i:i + batch_size]
        batch_images = []
        for path in batch_paths:
            image = np.load(path).astype(np.float32)
            image = torch.from_numpy(image)
            image = transforms(image)
            batch_images.append(image)

        images = torch.stack(batch_images).to(device)

        with torch.no_grad():
            embeddings = model.predict(images)

        sum_embedding += embeddings.cpu().numpy().sum(axis=0)
        count += len(images)

    del model
    torch.cuda.empty_cache()

    return sum_embedding, count


def worker_fn(rank, args, results):
    """Worker process for multi-GPU inference."""
    num_gpus = args.num_gpus
    device = f"cuda:{rank}"

    data_dir = args.data_dir
    npz_path = os.path.join(data_dir, 'ops_dataset.npz')
    data = np.load(npz_path, allow_pickle=True)
    total_samples = len(data['file_paths'])

    # Compute partition
    chunk_size = (total_samples + num_gpus - 1) // num_gpus
    start = rank * chunk_size
    end = min((rank + 1) * chunk_size, total_samples)
    indices = np.arange(start, end)

    print(f"[{device}] Processing indices {start:,} to {end:,} ({end - start:,} samples)")

    # Create partitioned dataset
    dataset = OPSImageDataset(data_dir, indices=indices, preprocessing=args.preprocessing)

    # Load encoder
    model = load_encoder(args.model_name, device)

    # Run inference
    suffix = f"_{args.preprocessing}" if args.preprocessing != "arcsinh" else ""
    output_path = os.path.join(data_dir, f'ops_embeddings{suffix}_gpu{rank}.npy')
    n_samples = run_inference_on_partition(model, dataset, args.batch_size, device, output_path)

    del model
    torch.cuda.empty_cache()

    # Partition NTC paths across GPUs
    ntc_paths = get_ntc_paths(data_dir)
    ntc_chunk_size = (len(ntc_paths) + num_gpus - 1) // num_gpus
    ntc_start = rank * ntc_chunk_size
    ntc_end = min((rank + 1) * ntc_chunk_size, len(ntc_paths))
    ntc_partition = ntc_paths[ntc_start:ntc_end]

    print(f"[{device}] Computing NTC embedding for {len(ntc_partition):,} images...")
    ntc_sum, ntc_count = compute_ntc_partial(
        ntc_partition, args.batch_size, device, args.model_name,
        preprocessing=args.preprocessing
    )

    # Store results
    results[rank] = {
        'n_samples': n_samples,
        'output_path': output_path,
        'ntc_sum': ntc_sum,
        'ntc_count': ntc_count,
    }


def run_multi_gpu_inference(args):
    """Run inference across multiple GPUs."""
    num_gpus = args.num_gpus
    data_dir = args.data_dir

    print(f"Running multi-GPU inference on {num_gpus} GPUs...")

    # Load dataset info
    npz_path = os.path.join(data_dir, 'ops_dataset.npz')
    data = np.load(npz_path, allow_pickle=True)
    total_samples = len(data['file_paths'])
    labels = data['perturbation_indices']

    # Shared results dict
    manager = mp.Manager()
    results = manager.dict()

    # Spawn workers
    mp.spawn(
        worker_fn,
        args=(args, results),
        nprocs=num_gpus,
        join=True,
    )

    # Combine embeddings
    print("\nCombining embeddings from all GPUs...")
    embed_dim = 384
    all_embeddings = np.zeros((total_samples, embed_dim), dtype=np.float32)

    total_ntc_sum = np.zeros(embed_dim, dtype=np.float64)
    total_ntc_count = 0

    for rank in range(num_gpus):
        r = results[rank]
        partial = np.load(r['output_path'])
        chunk_size = (total_samples + num_gpus - 1) // num_gpus
        start = rank * chunk_size
        end = min((rank + 1) * chunk_size, total_samples)
        all_embeddings[start:end] = partial
        total_ntc_sum += r['ntc_sum']
        total_ntc_count += r['ntc_count']

        # Clean up temp file
        os.remove(r['output_path'])

    # Save combined embeddings
    suffix = f"_{args.preprocessing}" if args.preprocessing != "arcsinh" else ""
    emb_path = os.path.join(data_dir, f'ops_embeddings{suffix}.npy')
    np.save(emb_path, all_embeddings)
    print(f"Saved combined embeddings: {emb_path} ({all_embeddings.shape})")

    # Compute NTC embedding
    ntc_embedding = total_ntc_sum / total_ntc_count
    print(f"NTC embedding computed from {total_ntc_count:,} images")

    return all_embeddings, labels, ntc_embedding


def run_single_gpu_inference(args):
    """Run inference on a single GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = args.data_dir
    suffix = f"_{args.preprocessing}" if args.preprocessing != "arcsinh" else ""

    # Load dataset
    dataset = OPSImageDataset(data_dir, preprocessing=args.preprocessing)

    # Load encoder
    model = load_encoder(args.model_name, device)

    # Run inference
    print(f"Running inference on {len(dataset):,} images...")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    embed_dim = 384
    n_samples = len(dataset)
    all_embeddings = np.zeros((n_samples, embed_dim), dtype=np.float32)

    idx = 0
    for images, labels in tqdm(loader, desc="Encoding"):
        images = images.to(device)

        with torch.no_grad():
            embeddings = model.predict(images)

        all_embeddings[idx:idx + len(images)] = embeddings.cpu().numpy()
        idx += len(images)

    del model
    torch.cuda.empty_cache()

    # Save embeddings
    emb_path = os.path.join(data_dir, f'ops_embeddings{suffix}.npy')
    np.save(emb_path, all_embeddings)
    print(f"Saved embeddings: {emb_path} ({all_embeddings.shape})")

    # Compute NTC embedding
    ntc_paths = get_ntc_paths(data_dir)
    if args.preprocessing == "arcsinh":
        ntc_transforms = T.Compose([
            T.Resize((256, 256)),
            Arcsinh(),
            T.Normalize(7., 7.)
        ])
    elif args.preprocessing == "minmax":
        ntc_transforms = T.Compose([
            T.Resize((256, 256)),
            MinMaxNormalize(),
        ])

    print(f"Computing NTC embedding from {len(ntc_paths):,} images...")
    ntc_sum = np.zeros(embed_dim, dtype=np.float64)
    ntc_count = 0

    for i in range(0, len(ntc_paths), args.batch_size):
        batch_paths = ntc_paths[i:i + args.batch_size]
        batch_images = []
        for path in batch_paths:
            image = np.load(path).astype(np.float32)
            image = torch.from_numpy(image)
            image = ntc_transforms(image)
            batch_images.append(image)

        images = torch.stack(batch_images).to(device)

        model = load_encoder(args.model_name, device)

        with torch.no_grad():
            embeddings = model.predict(images)

        ntc_sum += embeddings.cpu().numpy().sum(axis=0)
        ntc_count += len(images)

        del model
        torch.cuda.empty_cache()

    ntc_embedding = ntc_sum / ntc_count
    print(f"NTC embedding computed from {ntc_count:,} images")

    labels = dataset.perturbation_indices
    return all_embeddings, labels, ntc_embedding


def compute_class_means(all_embeddings, labels, num_classes, ntc_embedding):
    """Compute per-class mean embeddings, with NTC as last class."""
    print(f"Computing per-class means for {num_classes} perturbations...")

    embed_dim = all_embeddings.shape[1]
    class_means = np.zeros((num_classes + 1, embed_dim), dtype=np.float32)

    for c in tqdm(range(num_classes), desc="Class means"):
        mask = (labels == c)
        if mask.sum() > 0:
            class_means[c] = all_embeddings[mask].mean(axis=0)

    # Last class = NTC
    class_means[num_classes] = ntc_embedding

    return class_means


def main(args):
    data_dir = args.data_dir
    suffix = f"_{args.preprocessing}" if args.preprocessing != "arcsinh" else ""

    # Determine number of GPUs
    num_available = torch.cuda.device_count()
    if args.num_gpus == 'all':
        num_gpus = num_available
    else:
        num_gpus = int(args.num_gpus)

    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for parallel inference")
        args.num_gpus = num_gpus
        all_embeddings, labels, ntc_embedding = run_multi_gpu_inference(args)
    else:
        print(f"Using single GPU for inference")
        args.num_gpus = 1
        all_embeddings, labels, ntc_embedding = run_single_gpu_inference(args)

    # Compute per-class means
    num_classes = len(np.unique(labels))
    class_means = compute_class_means(all_embeddings, labels, num_classes, ntc_embedding)

    # Save class means
    means_path = os.path.join(data_dir, f'ops_class_means{suffix}.npy')
    np.save(means_path, class_means)
    print(f"Saved class means: {means_path} ({class_means.shape})")

    # Save metadata
    meta = {
        'num_classes': int(num_classes),
        'null_class_idx': int(num_classes),
        'embed_dim': int(all_embeddings.shape[1]),
        'total_samples': int(len(all_embeddings)),
        'ntc_samples': int(len(get_ntc_paths(data_dir))),
        'num_gpus_used': num_gpus,
        'preprocessing': args.preprocessing,
    }
    meta_path = os.path.join(data_dir, f'ops_encoder_meta{suffix}.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing ops_dataset.npz, ops_files.db')
    parser.add_argument('--model-name', type=str, default='recursionpharma/OpenPhenom',
                        help='HuggingFace model name for the encoder')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size per GPU for encoder inference')
    parser.add_argument('--num-gpus', type=str, default='all',
                        help='Number of GPUs to use (default: all, or specify an integer)')
    parser.add_argument('--preprocessing', type=str, default='arcsinh',
                        choices=['arcsinh', 'minmax'],
                        help='Preprocessing to apply before encoder (default: arcsinh)')
    args = parser.parse_args()
    main(args)
