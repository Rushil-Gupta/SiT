"""
Compute per-plate statistics for OPS images.

Reads funk22-npy.txt, groups images by plate ID (parts[1]_parts[2] of filename),
loads all images per plate, and calls get_stats() to produce per-plate stats.

Optionally visualizes samples from the first plate (raw + preprocessed)
to verify the pipeline.

Usage:
    python compute_plate_stats.py --data-dir /path/to/ops_data --vis-samples 10
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torchvision import transforms as T
from torchvision.utils import make_grid
from PIL import Image
from scipy.ndimage import gaussian_filter


class Arcsinh(nn.Module):
    def forward(self, x):
        return torch.arcsinh(x)


def get_plate_id(path):
    fname = os.path.basename(path).replace('.npy', '')
    parts = fname.split('_')
    return f"{parts[1]}_{parts[2]}"


def make_preprocess_vis(sample_paths, save_path, fields, p1, p99):
    """Save side-by-side grid of raw vs preprocessed images."""
    rng = np.random.RandomState(0)
    paths = rng.choice(sample_paths, size=min(len(sample_paths), 10), replace=False)

    raw = torch.stack([torch.from_numpy(np.load(p).astype(np.float32)) for p in paths])
    proc = np.clip(raw.clone().astype(torch.float32)/fields, 0, None)
    proc = np.clip((proc - p1) / (p99 - p1), 0, 1)
    # proc = preprocess(raw.clone())

    def to_grid(tensor, normalize=True, value_range=None):
        N, C, H, W = tensor.shape
        all_ch = tensor.unsqueeze(2).reshape(N * C, 1, H, W)
        kwargs = dict(nrow=4, normalize=normalize)
        if value_range is not None:
            kwargs['value_range'] = value_range
        grid = make_grid(all_ch, **kwargs)
        grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        return grid

    raw_grid = to_grid(raw, normalize=True)
    proc_grid = to_grid(proc, normalize=True, value_range=(-1, 1))

    gap = 10
    h, w = raw_grid.shape[:2]
    combined = np.ones((h, w * 2 + gap, 3), dtype=np.uint8) * 255
    combined[:h, :w] = raw_grid
    combined[:h, w + gap:] = proc_grid

    Image.fromarray(combined).save(save_path)
    print(f"  Visualization saved to {save_path}")

def get_flatfield(images, sigma=25):
    """
    images: (N, H, W) — all images from one plate+channel
    sigma: should be >> cell size in pixels. For 100x100 images, 
           try sigma=20-30 rather than 50 (which is for larger images)
    """
    mean_img = images.mean(axis=0)
    fields = np.zeros_like(mean_img)
    for c in range(mean_img.shape[1]):
        mean_ch = mean_img[c]
        flatfield = gaussian_filter(mean_ch, sigma=sigma)
        flatfield = flatfield / flatfield.mean()
        fields[c] = flatfield
        
    return fields
        
def get_stats(images):
    """
    Compute per-plate statistics on raw pixel values.

    Args:
        images: np.ndarray of shape (N, 4, 100, 100), float32

    Returns:
        tuple: (fair_field, dark_field, p1, p99)
    """
    flatfield = get_flatfield(images)

    p1s = np.zeros(images.shape[1])
    p99s = np.zeros(images.shape[1])
    for c in range(images.shape[1]):
        ch = np.clip(images[:, c].astype(np.float32) / flatfield[c], 0, None)  # Apply flatfield correction
        p1s[c] = np.percentile(ch, 1).astype(np.float32)
        p99s[c] = np.percentile(ch, 99).astype(np.float32)

    return flatfield, p1s, p99s


def main(args):
    txt_path = os.path.join(args.data_dir, 'funk22-npy.txt')
    with open(txt_path, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]

    plate_groups = defaultdict(list)
    for p in paths:
        plate_groups[get_plate_id(p)].append(p)

    print(f"Found {len(plate_groups)} plates, {len(paths)} total images")

    # preprocess = T.Compose([Arcsinh(), T.Normalize(7., 7.)])

    results = {}
    for i, (pid, group) in enumerate(sorted(plate_groups.items())):
        print(f"  {pid}  ({len(group)} images)")
        images = np.stack([np.load(p).astype(np.float32) for p in group])
        fields, p1, p99 = get_stats(images)

        if args.vis_samples > 0 and i == 0:
            os.makedirs(args.vis_dir, exist_ok=True)
            vis_path = os.path.join(args.vis_dir, f"plate_vis_{pid}.png")
            make_preprocess_vis(group, vis_path, fields,p1,p99)
            breakpoint()

        results[pid] = {
            "fields": fields,
            "p1": p1,
            "p99": p99,
            "n_images": len(group),
        }

    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} plates to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing funk22-npy.txt")
    parser.add_argument("--output-path", default="plate_stats.json",
                        help="Path to save the output JSON")
    parser.add_argument("--vis-samples", type=int, default=10,
                        help="Number of samples to visualize from first plate (0 to disable)")
    parser.add_argument("--vis-dir", default="results",
                        help="Directory to save visualizations")
    args = parser.parse_args()
    main(args)
