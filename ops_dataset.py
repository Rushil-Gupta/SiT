"""
PyTorch Dataset for OPS images with lazy loading.
"""
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class OPSDataset(Dataset):
    """
    Lazy-loading dataset for OPS perturbation images.

    Loads .npy files on-demand, applies per-channel standardization
    using precomputed nontargeting statistics.
    """

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory containing ops_dataset.npz, ops_perturbation_map.json, ops_stats.json
            transform: Optional transform to apply after standardization
        """
        self.data_dir = data_dir
        self.transform = transform

        # Load dataset arrays
        npz_path = os.path.join(data_dir, 'ops_dataset.npz')
        data = np.load(npz_path, allow_pickle=True)
        self.file_paths = data['file_paths']
        self.perturbation_indices = data['perturbation_indices']

        # Load perturbation map
        map_path = os.path.join(data_dir, 'ops_perturbation_map.json')
        with open(map_path, 'r') as f:
            raw_map = json.load(f)
        self.perturbation_map = {int(k): v for k, v in raw_map.items()}

        # Load stats and convert to tensors
        stats_path = os.path.join(data_dir, 'ops_stats.json')
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        self.mean = torch.tensor(stats['mean'], dtype=torch.float32).view(4, 1, 1)
        self.std = torch.tensor(stats['std'], dtype=torch.float32).view(4, 1, 1)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        perturbation_idx = int(self.perturbation_indices[idx])

        # Load image: (4, 100, 100), uint16
        image = np.load(path).astype(np.float32)

        # Convert to tensor
        image = torch.from_numpy(image)

        # Standardize: (x - mean) / std
        image = (image - self.mean) / self.std

        if self.transform is not None:
            image = self.transform(image)

        return image, perturbation_idx

    def get_perturbation_map(self):
        """Return dict mapping perturbation index to gene name."""
        return self.perturbation_map

    def get_num_perturbations(self):
        """Return number of unique perturbations."""
        return len(self.perturbation_map)
