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
    _ch_max = torch.Tensor([65535.0, 16628.0, 65212.0, 65535.0])  # Pre-computed per-channel max values from compute_data_range.py
    _ch_min = torch.Tensor([0.0, 0.0, 0.0, 0.0])  # Pre-computed per-channel min values from compute_data_range.py
    def forward(self, x):
        # return torch.arcsinh(x)
        if x.dim() != 4:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        # ch_min = x.view(x.shape[1], -1).min(dim=-1)[0]  #  (C)
        # ch_max = x.view(x.shape[1], -1).max(dim=-1)[0]  #  (C)
        ch_range = self._ch_max - self._ch_min

        x = (x - self._ch_min[None, :, None, None]) / ch_range[None, :, None, None]  # Scale to [0, 1]
        return x.squeeze(0)  # Remove batch dimension if it was added


class OPSDataset(Dataset):
    """
    Lazy-loading dataset for OPS perturbation images.

    Loads .npy files on-demand, applies arcsinh + normalize transforms.
    Supports subsetting perturbations and per-class imbalance.
    """

    def __init__(self, data_dir, max_perturbations=None, imbalance_factor=1.0, seed=0):
        """
        Args:
            data_dir: Directory containing ops_dataset.npz, ops_perturbation_map.json
            max_perturbations: If set, randomly select this many perturbations (from 1451).
            imbalance_factor: If < 1.0, randomly pick one class and drop (1-F) of its samples.
            seed: Random seed for reproducibility of subsetting and imbalance.
        """
        self.data_dir = data_dir

        # Load dataset arrays
        npz_path = os.path.join(data_dir, 'ops_dataset.npz')
        data = np.load(npz_path, allow_pickle=True)
        file_paths = list(data['file_paths'])
        perturbation_indices = np.array(data['perturbation_indices'], dtype=np.int64)

        # Load perturbation map
        map_path = os.path.join(data_dir, 'ops_perturbation_map.json')
        with open(map_path, 'r') as f:
            raw_map = json.load(f)
        self.perturbation_map = {int(k): v for k, v in raw_map.items()}

        total_perturbations = len(self.perturbation_map)  # 1451
        rng = np.random.RandomState(seed)

        # --- Perturbation subsetting ---
        if max_perturbations is not None and max_perturbations < total_perturbations:
            selected = sorted(rng.choice(total_perturbations, size=max_perturbations, replace=False))
            self.selected_original_indices = selected

            # Build mask and re-index
            old_to_new = {old: new for new, old in enumerate(selected)}
            mask = np.isin(perturbation_indices, selected)
            file_paths = [p for p, keep in zip(file_paths, mask) if keep]
            perturbation_indices = np.array([old_to_new[int(idx)] for idx in perturbation_indices[mask]], dtype=np.int64)

            # Filter perturbation map
            self.perturbation_map = {new: self.perturbation_map[old] for new, old in enumerate(selected)}
        else:
            self.selected_original_indices = list(range(total_perturbations))

        # --- Imbalance: drop samples from one random class ---
        if imbalance_factor < 1.0:
            num_selected = len(self.selected_original_indices)
            imbalanced_class = rng.randint(num_selected)

            # Find indices of this class in the filtered list
            class_mask = perturbation_indices == imbalanced_class
            class_indices = np.where(class_mask)[0]

            # Keep only imbalance_factor fraction
            rng.shuffle(class_indices)
            keep_count = max(1, int(imbalance_factor * len(class_indices)))
            drop_indices = set(class_indices[keep_count:])

            file_paths = [p for i, p in enumerate(file_paths) if i not in drop_indices]
            perturbation_indices = np.array([idx for i, idx in enumerate(perturbation_indices) if i not in drop_indices], dtype=np.int64)

        self.file_paths = file_paths
        self.perturbation_indices = perturbation_indices

        # Image transforms: arcsinh then normalize
        self.transforms = T.Compose([
            Arcsinh(),
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
