# SiT — Agent Notes

## Setup
- Conda env: `conda env create -f environment.yml && conda activate SiT`
- Requires GPU for training; `sample.py` works on CPU too
- Training needs ImageNet path: `--data-path /path/to/imagenet/train`

## Commands
- **Train** (DDP): `torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-XL/2 --data-path /path/to/imagenet/train`
- **Sample** (single GPU): `python sample.py ODE --image-size 256 --seed 1`
- **Sample** (multi-GPU, for FID): `torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py ODE --model SiT-XL/2 --num-fid-samples 50000`
- **WandB**: set `ENTITY` and `PROJECT` env vars, add `--wandb` flag to train command

## Architecture
- Flat single-file structure; no packages, no monorepo
- `models.py` — SiT model definitions (SiT_XL_2, SiT_L_4, etc.). Registry dict: `SiT_models`
- `train.py` — DDP training loop with EMA, checkpointing, optional wandb
- `sample.py` — single-GPU/CPU sampling, saves `sample.png`
- `sample_ddp.py` — multi-GPU sampling, outputs `.npz` for ADM evaluation suite
- `transport/` — interpolant/flow/sde math (path types: Linear, GVP, VP; predictions: velocity, score, noise)
- `train_utils.py` — CLI arg parsers for transport/ODE/SDE options
- `download.py` — auto-downloads pretrained weights to `pretrained_models/`
- `wandb_utils.py` — distributed-safe wandb logging (rank 0 only)

## Key conventions / gotchas
- Model names use `/` in CLI (`SiT-XL/2`) but filenames use `-` (`SiT-XL-2-256x256.pt`)
- Only `SiT-XL/2` at 256x256 has auto-download; other sizes/models require `--ckpt`
- VAE latent decode factor: `samples / 0.18215`
- Class 1000 = null class for classifier-free guidance
- Checkpoints from `train.py` contain keys: `model`, `ema`, `opt`, `args`
- Resuming training restores model+EMA+optimizer state and args from checkpoint
- ODE likelihood calculation requires `--cfg-scale 1` (incompatible with guidance)
- Default ODE solver: `dopri5` (via torchdiffeq); default SDE solver: `Euler`
- No tests, no lint, no typecheck — this is a research repo

## Outputs
- Training: `results/NNN-model-path-pred-weight/checkpoints/*.pt`
- Sampling: `samples/` directory with `.png` files and `.npz` for evaluation
- Gitignored: `samples/`, `results/`, `pretrained_models/`, `wandb/`, `*.out`
