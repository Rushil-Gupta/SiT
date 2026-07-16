"""
Test OPSDataset's generalized long-tailed class_distribution_file mechanism
(total_perturbations + tiers of imbalance factors -- see the OPSDataset
docstring in dataset/ops_dataset.py for the schema).

Usage:
    python test_scripts/test_class_distribution.py --data-dir /scratch/rg5218/SiT

Tests:
1. Multi-tier config (total_perturbations + tiers): correct class counts per
   tier, correct retained sample counts, untouched classes stay at full count.
2. total_perturbations only (no tiers): pure subsetting, no imbalance.
3. tiers only (no total_perturbations): falls back to --max-perturbations.
4. get_class_distribution_metadata(): grouping + gene names are correct.
"""
import argparse
import json
import os
import tempfile

import numpy as np


def _write_config(config):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(config, f)
    return path


def test_multi_tier(data_dir):
    from dataset import OPSDataset

    config = {
        "total_perturbations": 12,
        "tiers": {"0.1": 4, "0.05": 4, "0.01": 2},
    }
    config_path = _write_config(config)
    try:
        ds = OPSDataset(data_dir, class_distribution_file=config_path, seed=42)
        assert ds.get_num_perturbations() == 12, f"Expected 12, got {ds.get_num_perturbations()}"

        factors = ds.class_imbalance_factors
        assert len(factors) == 12
        tier_counts = {0.1: 0, 0.05: 0, 0.01: 0, 1.0: 0}
        for f in factors:
            tier_counts[round(float(f), 6)] += 1
        assert tier_counts[0.1] == 4, tier_counts
        assert tier_counts[0.05] == 4, tier_counts
        assert tier_counts[0.01] == 2, tier_counts
        assert tier_counts[1.0] == 2, tier_counts  # 12 - 4 - 4 - 2
        print(f"  [PASS] Tier class counts: {tier_counts}")

        # Per-class sample counts should roughly follow their assigned factor.
        counts = np.bincount(ds.perturbation_indices, minlength=12)
        # Recover original (pre-imbalance) per-class counts by re-running with
        # the same subsetting but no tiers, for comparison.
        subset_only_config = _write_config({"total_perturbations": 12})
        try:
            ds_full = OPSDataset(data_dir, class_distribution_file=subset_only_config, seed=42)
        finally:
            os.remove(subset_only_config)
        full_counts = np.bincount(ds_full.perturbation_indices, minlength=12)

        for class_idx, factor in enumerate(factors):
            expected = max(1, int(factor * full_counts[class_idx]))
            assert counts[class_idx] == expected, (
                f"class {class_idx} (factor={factor}): expected {expected}, got {counts[class_idx]}"
            )
        print(f"  [PASS] Retained sample counts match factor * original_count for every class")
    finally:
        os.remove(config_path)


def test_total_perturbations_only(data_dir):
    from dataset import OPSDataset

    config_path = _write_config({"total_perturbations": 8})
    try:
        ds = OPSDataset(data_dir, class_distribution_file=config_path, seed=1)
        assert ds.get_num_perturbations() == 8
        assert np.all(ds.class_imbalance_factors == 1.0), "No tiers given, all classes should be untouched"
        print(f"  [PASS] total_perturbations-only: {ds.get_num_perturbations()} classes, no imbalance applied")
    finally:
        os.remove(config_path)


def test_tiers_only_falls_back_to_max_perturbations(data_dir):
    from dataset import OPSDataset

    config_path = _write_config({"tiers": {"0.2": 2}})
    try:
        ds = OPSDataset(data_dir, max_perturbations=10, class_distribution_file=config_path, seed=1)
        assert ds.get_num_perturbations() == 10, "Should fall back to max_perturbations when total_perturbations is absent"
        tier_count = int((ds.class_imbalance_factors < 1.0).sum())
        assert tier_count == 2
        print(f"  [PASS] tiers-only config falls back to max_perturbations={10}")
    finally:
        os.remove(config_path)


def test_metadata(data_dir):
    from dataset import OPSDataset

    config = {
        "total_perturbations": 6,
        "tiers": {"0.1": 2},
    }
    config_path = _write_config(config)
    try:
        ds = OPSDataset(data_dir, class_distribution_file=config_path, seed=3)
        meta = ds.get_class_distribution_metadata(source_config_file=config_path)

        assert meta["source_config_file"] == config_path
        assert meta["total_perturbations"] == 6
        assert len(meta["no_imbalance"]) == 4
        assert len(meta["tiers"]["0.1"]) == 2

        for entry in meta["no_imbalance"] + meta["tiers"]["0.1"]:
            assert entry["gene"] == ds.perturbation_map[entry["class_idx"]]

        print(f"  [PASS] get_class_distribution_metadata: "
              f"{len(meta['no_imbalance'])} untouched, tiers={list(meta['tiers'].keys())}")
    finally:
        os.remove(config_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    print("=== Test 1: Multi-tier long-tailed config ===")
    test_multi_tier(args.data_dir)

    print("\n=== Test 2: total_perturbations only ===")
    test_total_perturbations_only(args.data_dir)

    print("\n=== Test 3: tiers only (falls back to max_perturbations) ===")
    test_tiers_only_falls_back_to_max_perturbations(args.data_dir)

    print("\n=== Test 4: get_class_distribution_metadata ===")
    test_metadata(args.data_dir)

    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    main()
