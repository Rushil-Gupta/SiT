"""
Test the OPSDataset subsampling and imbalance logic.

Usage:
    python test_subsampling.py --data-dir /scratch/rg5218/SiT

Tests:
1. Default (max_perturbations=None, no class_distribution_file)
2. Subset only (max_perturbations=10)
3. Imbalance only (single-tier class_distribution_file, factor 0.3)
4. Both (max_perturbations=10, single-tier class_distribution_file, factor 0.3)
5. Class means slicing matches the selected indices

See test_class_distribution.py for full multi-tier long-tailed coverage.
"""
import argparse
import json
import tempfile
import os
import numpy as np


def _write_single_tier_config(factor, num_classes=1):
    """Writes a one-tier class_distribution_file reproducing the old
    single-class imbalance_factor behavior, for reuse by tests below."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({"tiers": {str(factor): num_classes}}, f)
    return path


def test_default(data_dir):
    from dataset import OPSDataset
    ds = OPSDataset(data_dir)
    assert ds.get_num_perturbations() == 1451, f"Expected 1451, got {ds.get_num_perturbations()}"
    assert len(ds.selected_original_indices) == 1451
    print(f"  [PASS] Default: {len(ds)} images, {ds.get_num_perturbations()} classes")


def test_subset(data_dir):
    from dataset import OPSDataset
    ds = OPSDataset(data_dir, max_perturbations=10, seed=42)
    assert ds.get_num_perturbations() == 10, f"Expected 10, got {ds.get_num_perturbations()}"
    assert len(ds.selected_original_indices) == 10
    # All indices should be within 0..9 (re-indexed)
    unique = set(ds.perturbation_indices)
    assert unique == set(range(10)), f"Re-indexing wrong: {unique}"
    print(f"  [PASS] Subset(10): {len(ds)} images, {ds.get_num_perturbations()} classes, "
          f"original indices: {ds.selected_original_indices}")

    # Same seed should give same result
    ds2 = OPSDataset(data_dir, max_perturbations=10, seed=42)
    assert ds.selected_original_indices == ds2.selected_original_indices
    print(f"  [PASS] Subset reproducibility with seed=42")


def test_imbalance(data_dir):
    from dataset import OPSDataset
    config_path = _write_single_tier_config(0.3)
    try:
        ds = OPSDataset(data_dir, class_distribution_file=config_path, seed=42)
        assert ds.get_num_perturbations() == 1451

        # Count samples per class
        counts = np.bincount(ds.perturbation_indices)
        min_count = counts.min()
        max_count = counts.max()

        # With imbalance, min should be noticeably smaller than max
        # (one class lost ~70% of its samples)
        print(f"  Per-class counts: min={min_count}, max={max_count}, mean={counts.mean():.1f}")
        assert min_count < max_count, "Imbalance did not reduce any class"
        assert max_count / min_count > 2.0, f"Imbalance too mild: {max_count}/{min_count}"
        print(f"  [PASS] Imbalance(0.3): {len(ds)} images, counts min={min_count} max={max_count}")

        # Reproducibility
        ds2 = OPSDataset(data_dir, class_distribution_file=config_path, seed=42)
        np.testing.assert_array_equal(ds.perturbation_indices, ds2.perturbation_indices)
        print(f"  [PASS] Imbalance reproducibility with seed=42")
    finally:
        os.remove(config_path)


def test_both(data_dir):
    from dataset import OPSDataset
    config_path = _write_single_tier_config(0.3)
    try:
        ds = OPSDataset(data_dir, max_perturbations=20, class_distribution_file=config_path, seed=42)
        assert ds.get_num_perturbations() == 20
        unique = set(ds.perturbation_indices)
        assert unique == set(range(20)), f"Re-indexing wrong after imbalance"

        counts = np.bincount(ds.perturbation_indices, minlength=20)
        print(f"  Subset(20)+Imbalance(0.3): {len(ds)} images, counts: min={counts.min()} max={counts.max()}")
        assert counts.min() < counts.max()
        print(f"  [PASS] Subset + Imbalance")
    finally:
        os.remove(config_path)


def test_class_means_slice(data_dir):
    from dataset import OPSDataset
    ds = OPSDataset(data_dir, max_perturbations=15, seed=7)
    selected = ds.selected_original_indices

    # Load full class means
    full = np.load(f"{data_dir}/ops_class_means.npy")  # (1452, D)
    sliced = np.concatenate([full[selected], full[-1:]], axis=0)

    assert sliced.shape[0] == 16, f"Expected 16 (15 pert + null), got {sliced.shape[0]}"
    # Check that the null class is the last row
    np.testing.assert_array_equal(sliced[-1], full[-1])
    print(f"  [PASS] Class means slice: {sliced.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    print("=== Test 1: Default dataset ===")
    test_default(args.data_dir)

    print("\n=== Test 2: Subset perturbations ===")
    test_subset(args.data_dir)

    print("\n=== Test 3: Imbalance ===")
    test_imbalance(args.data_dir)

    print("\n=== Test 4: Subset + Imbalance together ===")
    test_both(args.data_dir)

    print("\n=== Test 5: Class means slicing ===")
    test_class_means_slice(args.data_dir)

    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    main()
