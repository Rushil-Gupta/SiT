"""
Reconstruction quality check for FunkVAE (see vae.py).

Runs a batch of real OPS images through pad->encode->decode->crop and reports
per-channel MSE against the originals. Run once against the frozen,
channel-surgery-only VAE (no --vae-checkpoint) as a baseline, and again after
finetune_vae.py has produced a checkpoint, to confirm finetuning actually
improves reconstruction.

Usage:
    python test_scripts/test_funk_vae_reconstruction.py --ops-data-dir /scratch/rg5218/SiT
    python test_scripts/test_funk_vae_reconstruction.py --ops-data-dir /scratch/rg5218/SiT --vae-checkpoint <path>
"""
import argparse
import os

import torch

from dataset import OPSDataset
from vae import FunkVAE


def to_pm1(x: torch.Tensor) -> torch.Tensor:
    return x * 2 - 1


def test_reconstruction(data_dir, vae_checkpoint, batch_size, device):
    dataset = OPSDataset(data_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    x, _ = next(iter(loader))
    x = to_pm1(x.to(device))

    vae = FunkVAE(checkpoint_path=vae_checkpoint, device=device)
    vae.eval()

    with torch.no_grad():
        x_padded = vae.pad(x)
        assert x_padded.shape[-2:] == (FunkVAE.PAD_TARGET, FunkVAE.PAD_TARGET), \
            f"Expected padded shape {FunkVAE.PAD_TARGET}, got {x_padded.shape[-2:]}"

        z, posterior = vae.encode(x_padded, sample=False)
        assert z.shape[1] == FunkVAE.NUM_CHANNELS
        print(f"  Latent shape: {tuple(z.shape)}")

        recon = vae.decode(z, crop_size=x.shape[-1])
        assert recon.shape == x.shape, f"Shape mismatch: recon={recon.shape}, x={x.shape}"

        per_channel_mse = ((recon - x) ** 2).mean(dim=(0, 2, 3))
        overall_mse = ((recon - x) ** 2).mean()

    print(f"  Overall MSE: {overall_mse.item():.6f}")
    for c, mse in enumerate(per_channel_mse.tolist()):
        print(f"  Channel {c} MSE: {mse:.6f}")

    print("  [PASS] Reconstruction ran end-to-end with matching shapes.")
    return per_channel_mse.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops-data-dir", required=True)
    parser.add_argument("--dataset", default="funk")
    parser.add_argument("--vae-checkpoint", default=None,
                         help="Omit to test the frozen channel-surgery-only VAE (no finetuning).")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = os.path.join(args.ops_data_dir, args.dataset)

    print(f"=== FunkVAE reconstruction test (checkpoint={args.vae_checkpoint}) ===")
    test_reconstruction(data_dir, args.vae_checkpoint, args.batch_size, device)


if __name__ == "__main__":
    main()
