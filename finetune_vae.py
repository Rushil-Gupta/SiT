"""
Finetunes FunkVAE (see vae.py) -- the channel-adapted Stable Diffusion VAE --
on 4-channel OPS microscopy images, using PyTorch DDP (this dataset is
1.1M+ images, the same scale train.py already runs under DDP -- see
dataset/funk/compute_data_range.py).

Both the encoder and decoder are finetuned (not just the decoder): the
pretrained encoder's filters were learned on natural RGB photos and have no
built-in reason to transfer to punctate, largely textureless, single-stain-
per-channel fluorescence microscopy data, so freezing it would handicap
reconstruction quality for no real savings.

Loss is reconstruction (L1) + a small-weight KL term on the latent
distribution. LPIPS perceptual loss is intentionally not used since it's a
VGG-based metric defined for 3-channel RGB and doesn't transparently apply to
our 4 independent microscopy channels.
"""
import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from dataset import OPSDataset
from vae import FunkVAE


def to_pm1(x: torch.Tensor) -> torch.Tensor:
    """OPSDataset yields images min-max normalized to [0, 1]; VAE expects [-1, 1]."""
    return x * 2 - 1


@torch.no_grad()
def per_channel_recon_error(recon: torch.Tensor, target: torch.Tensor, device):
    """Per-channel MSE, summed (not averaged) across ranks so the caller can
    divide by the true total sample count once all ranks' contributions are in."""
    se = ((recon - target) ** 2).sum(dim=(0, 2, 3))
    n = torch.tensor(recon.shape[0] * recon.shape[2] * recon.shape[3], device=device, dtype=se.dtype)
    dist.all_reduce(se, op=dist.ReduceOp.SUM)
    dist.all_reduce(n, op=dist.ReduceOp.SUM)
    return (se / n).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops-data-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="funk")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Finetuning currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    world_size = dist.get_world_size()
    local_batch_size = int(args.global_batch_size // world_size)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Starting rank={rank}, world_size={world_size}, local_batch_size={local_batch_size}")

    data_dir = os.path.join(args.ops_data_dir, args.dataset)
    dataset = OPSDataset(data_dir)
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    split_gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_gen)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=local_batch_size, shuffle=False, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=local_batch_size, shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    vae = FunkVAE(device=device)
    vae.train()
    vae = DDP(vae, device_ids=[device])

    opt = torch.optim.AdamW(vae.parameters(), lr=args.learning_rate)

    step = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        for x, _ in train_loader:
            x = to_pm1(x.to(device))
            x_padded = vae.module.pad(x)

            z, posterior = vae.module.encode(x_padded, sample=True)
            recon = vae.module.decode(z, crop_size=x.shape[-1])

            recon_loss = F.l1_loss(recon, x)
            kl_loss = posterior.kl().mean()
            loss = recon_loss + args.kl_weight * kl_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

            if step % args.log_every == 0 and rank == 0:
                print(f"[step {step}] epoch={epoch} recon_loss={recon_loss.item():.4f} "
                      f"kl_loss={kl_loss.item():.4f} loss={loss.item():.4f}")

            if step % args.ckpt_every == 0:
                vae.eval()
                with torch.no_grad():
                    val_x, _ = next(iter(val_loader))
                    val_x = to_pm1(val_x.to(device))
                    val_padded = vae.module.pad(val_x)
                    val_z, _ = vae.module.encode(val_padded, sample=False)
                    val_recon = vae.module.decode(val_z, crop_size=val_x.shape[-1])
                    val_mse = per_channel_recon_error(val_recon, val_x, device)
                    if rank == 0:
                        print(f"[step {step}] val per-channel MSE: {val_mse}")
                vae.train()

                if rank == 0:
                    ckpt_path = os.path.join(args.output_dir, f"vae_{step:07d}.pt")
                    torch.save({"vae": vae.module.state_dict(), "step": step, "args": vars(args)}, ckpt_path)
                    print(f"Saved checkpoint to {ckpt_path}")
                dist.barrier()

    if rank == 0:
        final_path = os.path.join(args.output_dir, "vae_final.pt")
        torch.save({"vae": vae.module.state_dict(), "step": step, "args": vars(args)}, final_path)
        print(f"Saved final checkpoint to {final_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
