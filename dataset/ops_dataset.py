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


class GlobalMinMaxNorm(nn.Module):
    _ch_max = torch.Tensor([65535.0, 16628.0, 65212.0, 65535.0])  # Pre-computed per-channel max values from compute_data_range.py
    _ch_min = torch.Tensor([0.0, 0.0, 0.0, 0.0])  # Pre-computed per-channel min values from compute_data_range.py
    def forward(self, x):
        self._ch_max = self._ch_max.to(x.device)
        self._ch_min = self._ch_min.to(x.device)
        n_dims = x.dim()
        if n_dims != 4:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        ch_range = self._ch_max - self._ch_min

        x = (x - self._ch_min[None, :, None, None]) / ch_range[None, :, None, None]  # Scale to [0, 1]
        if n_dims !=4:
            x = x.squeeze(0)  # Remove batch dimension if it was added
        return x


class OPSDataset(Dataset):
    """
    Lazy-loading dataset for OPS perturbation images.

    Loads .npy files on-demand, applies arcsinh + normalize transforms.
    Supports subsetting perturbations and per-class imbalance.
    """

    def __init__(self, data_dir, max_perturbations=None, class_distribution_file=None, seed=0):
        """
        Args:
            data_dir: Directory containing ops_dataset.npz, ops_perturbation_map.json
            max_perturbations: If set, randomly select this many perturbations (from 1451).
                Ignored if class_distribution_file sets "total_perturbations".
            class_distribution_file: Optional path to a JSON file describing the
                class-distribution scenario for this run:
                    {
                      "total_perturbations": 12,
                      "tiers": {"0.1": 4, "0.05": 4, "0.01": 2}
                    }
                "total_perturbations" (optional) overrides max_perturbations for
                subsetting. "tiers" (optional) maps an imbalance factor to a
                number of classes to assign to that tier -- each such class has
                (1-factor) of its samples dropped. Classes not covered by any
                tier keep their full sample count (factor 1.0). Both keys are
                independently optional.
            seed: Random seed for reproducibility of subsetting and imbalance.
        """
        self.data_dir = data_dir
        self.class_distribution_file = class_distribution_file

        class_distribution_config = {}
        if class_distribution_file is not None:
            with open(class_distribution_file, 'r') as f:
                class_distribution_config = json.load(f)
            if "total_perturbations" in class_distribution_config:
                max_perturbations = class_distribution_config["total_perturbations"]

        # Load dataset arrays
        npz_path = os.path.join(data_dir, 'ops_dataset.npz')
        data = np.load(npz_path, allow_pickle=True)
        file_paths = list(data['file_paths'])
        perturbation_indices = np.array(data['perturbation_indices'], dtype=np.int64)

        # Tracks each retained sample's row position in the original, unfiltered
        # file_paths list, so a cached array keyed by that original order (e.g.
        # latents.npy, see dataset/funk/precompute_latents.py) can still be
        # looked up correctly after subsetting/imbalance filtering below.
        sample_indices = np.arange(len(file_paths))

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
            sample_indices = sample_indices[mask]
            perturbation_indices = np.array([old_to_new[int(idx)] for idx in perturbation_indices[mask]], dtype=np.int64)

            # Filter perturbation map
            self.perturbation_map = {new: self.perturbation_map[old] for new, old in enumerate(selected)}
        else:
            self.selected_original_indices = list(range(total_perturbations))

        # --- Imbalance: drop samples from tiered classes (long-tailed setting) ---
        num_selected = len(self.selected_original_indices)
        self.class_imbalance_factors = np.ones(num_selected)
        tiers = class_distribution_config.get("tiers")
        if tiers:
            tier_items = [(float(factor), int(count)) for factor, count in tiers.items()]
            total_tiered = sum(count for _, count in tier_items)
            assert total_tiered <= num_selected, (
                f"class_distribution_file tiers request {total_tiered} classes, "
                f"but only {num_selected} classes are available after subsetting."
            )

            shuffled_classes = rng.permutation(num_selected)
            cursor = 0
            all_drop_indices = set()
            for factor, count in tier_items:
                tier_classes = shuffled_classes[cursor:cursor + count]
                cursor += count
                self.class_imbalance_factors[tier_classes] = factor

                for tier_class in tier_classes:
                    # Find indices of this class in the filtered list
                    class_mask = perturbation_indices == tier_class
                    class_indices = np.where(class_mask)[0]

                    # Keep only `factor` fraction of this class's samples
                    rng.shuffle(class_indices)
                    keep_count = max(1, int(factor * len(class_indices)))
                    all_drop_indices.update(class_indices[keep_count:])

            file_paths = [p for i, p in enumerate(file_paths) if i not in all_drop_indices]
            perturbation_indices = np.array(
                [idx for i, idx in enumerate(perturbation_indices) if i not in all_drop_indices], dtype=np.int64
            )
            keep_mask = np.array([i not in all_drop_indices for i in range(len(sample_indices))], dtype=bool)
            sample_indices = sample_indices[keep_mask]

        self.file_paths = file_paths
        self.perturbation_indices = perturbation_indices
        self.sample_indices = sample_indices

        # Image transforms: arcsinh then normalize
        self.transforms = T.Compose([
            GlobalMinMaxNorm(),
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

    def get_class_distribution_metadata(self, source_config_file=None):
        """
        Returns a JSON-serializable dict describing which classes ended up in
        which imbalance tier, with gene names attached (via perturbation_map,
        already re-keyed to this dataset's compact class-index space) -- useful
        metadata to persist alongside a training run. No disk I/O here; the
        caller decides where to write it (e.g. train.py writes it into the
        experiment directory).
        """
        no_imbalance = []
        tiers = {}
        for class_idx, factor in enumerate(self.class_imbalance_factors):
            entry = {"class_idx": int(class_idx), "gene": self.perturbation_map[class_idx]}
            if factor >= 1.0:
                no_imbalance.append(entry)
            else:
                tiers.setdefault(f"{factor:g}", []).append(entry)

        return {
            "source_config_file": source_config_file,
            "total_perturbations": len(self.perturbation_map),
            "no_imbalance": no_imbalance,
            "tiers": tiers,
        }


class OPSLatentDataset(OPSDataset):
    """
    Like OPSDataset, but loads precomputed VAE latents from a single packed
    array (dataset/funk/precompute_latents.py's latents.npy) instead of raw
    per-sample image files.

    Reuses OPSDataset's perturbation subsetting/imbalance logic via
    super().__init__() (which also builds self.sample_indices, each retained
    sample's row position in the original unfiltered file_paths order -- the
    same order latents.npy was written in), then looks up latents by that
    index and applies the empirical latent scale factor instead of
    GlobalMinMaxNorm.
    """

    def __init__(self, data_dir, latent_dir, max_perturbations=None, class_distribution_file=None, seed=0):
        super().__init__(data_dir, max_perturbations=max_perturbations,
                          class_distribution_file=class_distribution_file, seed=seed)
        self.latent_dir = latent_dir
        with open(os.path.join(latent_dir, "latent_meta.json"), "r") as f:
            meta = json.load(f)
        self.scale_factor = meta["scale_factor"]
        self.latent_shape = tuple(meta["latent_shape"])
        # Opened lazily per-worker process in __getitem__, not here: DataLoader
        # workers are separate processes, and opening a memmap before they
        # fork/spawn can behave inconsistently across platforms.
        self._latents = None

    def _get_latents(self):
        if self._latents is None:
            self._latents = np.load(os.path.join(self.latent_dir, "latents.npy"), mmap_mode="r")
        return self._latents

    def __getitem__(self, idx):
        perturbation_idx = int(self.perturbation_indices[idx])
        row = int(self.sample_indices[idx])

        latent = np.array(self._get_latents()[row], dtype=np.float32)
        latent = torch.from_numpy(latent) * self.scale_factor

        return latent, perturbation_idx
