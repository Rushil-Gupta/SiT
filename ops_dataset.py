"""
PyTorch Dataset for OPS images with lazy loading.
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms as T


class Arcsinh(nn.Module):
    def forward(self, x):
        return torch.arcsinh(x)


class OPSDataset(Dataset):
    """
    Lazy-loading dataset for OPS perturbation images.

    Loads .npy files on-demand, applies arcsinh + normalize transforms.
    """

    def __init__(self, data_dir):
        """
        Args:
            data_dir: Directory containing ops_dataset.npz, ops_perturbation_map.json
        """
        self.data_dir = data_dir

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

        # Image transforms: arcsinh then normalize
        self.transforms = T.Compose([
            Arcsinh(),
            T.Normalize(7., 7.)
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        perturbation_idx = int(self.perturbation_indices[idx])

        # Load image: (4, 100, 100), uint16
        image = np.load(path).astype(np.float32)

        # Convert to tensor and apply transforms
        image = torch.from_numpy(image)
        image = self.transforms(image)

        return image, perturbation_idx

    def get_perturbation_map(self):
        """Return dict mapping perturbation index to gene name."""
        return self.perturbation_map

    def get_num_perturbations(self):
        """Return number of unique perturbations."""
        return len(self.perturbation_map)
