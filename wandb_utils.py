import wandb
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
import os
import argparse
import hashlib


def is_main_process():
    return dist.get_rank() == 0


def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(args, exp_name, entity=None, project=None, rid=None, resume=None):
    config_dict = namespace_to_dict(args)
    wandb.login(key=os.environ["WANDB_KEY"])
    if rid is None:
        run_id = generate_run_id(exp_name)
        res = "allow"
    else:
        run_id = rid
        res = resume
    wandb.init(
        entity=entity,
        project=project,
        name=exp_name,
        config=config_dict,
        id=run_id,
        resume=res,
    )


def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(sample, step, save_dir):
    if not is_main_process():
        return

    sample = torch.clamp(sample, -1.0, 1.0)  # Ensure sample is in [-1, 1]
    N, C, H, W = sample.shape
    rows_per_page = 32

    vis_path = os.path.join(save_dir, f"step_{step:07d}")
    os.makedirs(vis_path, exist_ok=True)

    from PIL import Image

    for page_start in range(0, N, rows_per_page):
        chunk = sample[page_start : page_start + rows_per_page]
        K = chunk.size(0)

        # (K, C, H, W) -> (K*C, 1, H, W), grid with nrow=C → rows=samples, cols=channels
        all_channels = chunk.unsqueeze(2).reshape(K * C, 1, H, W)
        grid = make_grid(all_channels, nrow=C, normalize=True)
        grid_np = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        # WandB
        if wandb.run is not None:
            wandb.log({f"samples/page_{page_start // rows_per_page}": wandb.Image(grid_np)}, step=step)

        # Local save
        Image.fromarray(grid_np).save(os.path.join(vis_path, f"samples_page_{page_start // rows_per_page}.png"))
