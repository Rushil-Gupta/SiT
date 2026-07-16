"""
Sanity check for the metrics/ extractor registry: instantiates every
registered feature extractor and runs a dummy batch through .encode() in
both gen=True (generated-sample) and gen=False (real-sample) modes.

Catches: the registry never being populated (metrics/extractors not
imported), a missing `gen` kwarg on _preprocess, and 4->3 channel-mismatch
crashes -- all without needing real data or a trained checkpoint.

Usage:
    python test_scripts/test_extractors.py
"""
import torch

from metrics.registry import list_extractors, get_extractor


def test_extractor(name, device):
    print(f"--- Testing extractor: {name} ---")
    extractor = get_extractor(name, device=device)

    batch = torch.rand(2, 4, 100, 100, device=device)
    for gen in (True, False):
        feats = extractor.encode(batch, gen=gen)
        print(f"  gen={gen}: output shape {tuple(feats.shape)}")
        assert feats.shape == (2, extractor.dim), \
            f"Expected shape (2, {extractor.dim}), got {tuple(feats.shape)}"
        assert not torch.isnan(feats).any(), f"{name} produced NaNs (gen={gen})"

    print(f"  [PASS] {name}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    names = list_extractors()
    assert len(names) > 0, "No extractors registered -- metrics/extractors import is broken"
    print(f"Registered extractors: {names}")

    for name in names:
        test_extractor(name, device)

    print("\n=== All extractors passed! ===")


if __name__ == "__main__":
    main()
