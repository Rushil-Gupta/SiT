# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import json
import re
from pathlib import Path

from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from train_utils import parse_transport_args
from ops_dataset import OPSDataset
import wandb_utils
from PIL import Image
from torchvision.utils import make_grid


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def get_beta_schedule(step, num_warmup_steps, num_steps, beta_start=0.0, beta_end=0.1):
    """
    linear schedule for beta (KL weight) from beta_start to beta_end over num_steps, with a warmup period of num_warmup_steps
    """
    if step < num_warmup_steps:
        return beta_start
    else:
        progress = (step - num_warmup_steps) / max(1, num_steps - num_warmup_steps)
        return beta_start + progress * (beta_end - beta_start)


def make_channel_grid(images, nrow, normalize=True, value_range=None): 
    N, C, H, W = images.shape
    all_ch = images.unsqueeze(2).reshape(N * C, 1, H, W)
    kwargs = dict(nrow=nrow, normalize=normalize)
    if value_range is not None:
        kwargs['value_range'] = value_range
    grid = make_grid(all_ch, **kwargs)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return grid


@torch.no_grad()
def compute_val_loss(model, transport, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        model_kwargs = dict(y=y, beta=0.0)
        loss_dict = transport.training_losses(model, x, model_kwargs)
        per_sample = loss_dict["loss"]
        total_loss += per_sample.sum().item()
        total_samples += per_sample.size(0)
    model.train()
    # Reduce across GPUs
    total_loss_t = torch.tensor(total_loss, device=device)
    total_samples_t = torch.tensor(total_samples, device=device)
    dist.all_reduce(total_loss_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_t, op=dist.ReduceOp.SUM)
    return (total_loss_t / total_samples_t).item()


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                        f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        if args.ckpt is not None:
            experiment_name = Path(args.ckpt).parents[1].name
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)

        ###DUMP THE ARGS TO A JSON FILE
        with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        ## WANDB initialization (rank 0 only):
        init_kwargs = {
            "entity": os.environ["ENTITY"],
            "project": os.environ["PROJECT"],
        }
        if args.ckpt is not None:
            init_kwargs["rid"] = "97956759"
            init_kwargs["resume"] = "must"
        if args.wandb:
            wandb_utils.initialize(args, experiment_name, **init_kwargs)
    else:
        logger = create_logger(None)

    # Setup data:
    dataset = OPSDataset(
        args.ops_data_dir,
        max_perturbations=args.max_perturbations,
        imbalance_factor=args.imbalance_factor,
        seed=args.global_seed,
    )
    num_classes = dataset.get_num_perturbations()
    perturbation_map = dataset.get_perturbation_map() #Mapping from perturbation index to gene name

    # Split into train / val
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    split_gen = torch.Generator().manual_seed(args.global_seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=split_gen
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    logger.info(f"Dataset contains {len(dataset):,} images ({train_size} train, {val_size} val) across {num_classes} perturbations")
    if rank == 0:
        logger.info(f"Perturbation map (first 10): {dict(list(perturbation_map.items())[:10])}")

    # Create model:
    input_size = args.image_size #4 x 100 x 100 for the image data

    """
    We use SiT-S-2 with depth = 12, heads = 6 and patch size 2
    """
    model = SiT_models[args.model](
        input_size=input_size,
        num_classes=num_classes, #1451 perturbations + 1 control
        use_guidance=not args.use_direct_embed, #Direct embedding is using the mean embeddings from the frozen model
        use_direct_embed=args.use_direct_embed,
        embed_dim=args.embed_dim, #384 in our experiments,
        in_channels=4, #4 channels in the image data
        class_dropout_prob=args.class_dropout
    )

    # Load class means into embedding module:
    # Class means obtained offline by encoding all images of a given perturbation using frozen encoder
    # and averaging the resulting embeddings. See generate_embeddings.py for details.
    if args.use_direct_embed or args.use_guidance:
        class_means_path = os.path.join(args.ops_data_dir, 'ops_class_means.npy')
        assert os.path.isfile(class_means_path), f"Class means not found: {class_means_path}"
        full_means = np.load(class_means_path)  # (1452, D): 1451 pert + null
        # Slice to selected perturbations + null
        selected = dataset.selected_original_indices
        class_means = np.concatenate([full_means[selected], full_means[-1:]], axis=0)
        model.y_embedder.class_means.copy_(torch.from_numpy(class_means))
        logger.info(f"Loaded class means ({len(selected)} pert + null) from {class_means_path}")

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    model = DDP(model.to(device), device_ids=[device]) #Copy of model on each GPU

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    
    # Best checkpoint tracking (defined early so checkpoint resume can reference them)
    best_val_loss = float("inf")
    best_step = 0
    best_epoch = 0

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_recon_loss = 0
    running_kl_loss = 0

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"])
        opt.load_state_dict(state_dict["opt"])
        args = state_dict["args"]

        match = re.search(r'\d+', args.ckpt)
        train_steps = int(match.group()) if match else 0
        # Restore best tracking if resuming from a checkpoint that has it
        if "best_val_loss" in state_dict:
            best_val_loss = state_dict["best_val_loss"]
            best_step = state_dict["best_step"]
            best_epoch = state_dict["best_epoch"]
            logger.info(f"Resumed best tracking: best_val_loss={best_val_loss:.6f} at step={best_step} epoch={best_epoch}")

    requires_grad(ema, False) #Set EMA model params to not require gradients since we won't be optimizing the EMA model directly

    # Creates a Transport Object with Linear Path, Velocity Prediction and train, sample_eps=0
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity;

    #Sampler object for sampling with the transport conditions and hyperparameters specified in the args
    transport_sampler = Sampler(transport)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    # train_steps = 0
    # log_steps = 0
    # running_loss = 0
    # running_recon_loss = 0
    # running_kl_loss = 0
    start_time = time()

    # Labels to condition the model with (feel free to change):
    ys = torch.randint(num_classes, size=(local_batch_size,), device=device)
    use_cfg = args.cfg_scale > 1.0
    # Create sampling noise:
    n = ys.size(0)
    zs = torch.randn(n, 4, input_size, input_size, device=device)

    # Setup classifier-free guidance:
    if use_cfg: #NOT USED! CFG used directly in the embedding guidance module
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([num_classes] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys, beta=0.0)
        model_fn = ema.forward

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch) #DistributedSampler
        logger.info(f"Beginning epoch {epoch}...")
        # dist.breakpoint()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            
            beta = 0.0#get_beta_schedule(train_steps, num_warmup_steps=10000, num_steps=args.epochs * len(train_loader), beta_start=0.0, beta_end=args.beta)
            model_kwargs = dict(y=y, beta=beta)
            
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean() #Mean along the batch dimension

            # Compute KL loss for guidance module
            
            if args.use_guidance and not args.use_direct_embed and beta > 0:
                kl_loss = model.module._kl_loss
                if kl_loss is not None:
                    loss = loss + beta * kl_loss
                else:
                    kl_loss = torch.tensor(0.0, device=device)
            else:
                kl_loss = torch.tensor(0.0, device=device)

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            running_recon_loss += loss_dict["loss"].mean().item()
            running_kl_loss += kl_loss.item() if kl_loss.numel() > 0 else 0.0
            log_steps += 1
            train_steps += 1

            # Update empirical Bayes prior
            # if (args.use_guidance and not args.use_direct_embed
            #     and args.empirical_bayes_update_freq > 0
            #     and train_steps % args.empirical_bayes_update_freq == 0):
            #     model.module.y_embedder.update_empirical_bayes()
            #     dist.barrier()  # Ensure all processes update the prior before next training step
                # dist.breakpoint()

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_recon_loss = torch.tensor(running_recon_loss / log_steps, device=device)
                avg_kl_loss = torch.tensor(running_kl_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_recon_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_kl_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_recon_loss = avg_recon_loss.item() / dist.get_world_size()
                avg_kl_loss = avg_kl_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        { "train loss": avg_loss, "recon loss": avg_recon_loss, "kl loss": avg_kl_loss, "train steps/sec": steps_per_sec },
                        step=train_steps
                    )
                    # dist.breakpoint()
                # Reset monitoring variables:
                running_loss = 0
                running_recon_loss = 0
                running_kl_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint + validate:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

                # Compute validation loss (all ranks participate)
                val_loss = compute_val_loss(model, transport, val_loader, device)
                if rank == 0:
                    logger.info(f"(step={train_steps:07d}) Val Loss: {val_loss:.6f}")
                    if args.wandb:
                        wandb_utils.log({"val loss": val_loss}, step=train_steps)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_step = train_steps
                        best_epoch = epoch
                        best_ckpt = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args,
                            "best_val_loss": best_val_loss,
                            "best_step": best_step,
                            "best_epoch": best_epoch,
                        }
                        torch.save(best_ckpt, f"{checkpoint_dir}/best.pt")
                        with open(f"{checkpoint_dir}/best_info.json", "w") as f:
                            json.dump({"step": best_step, "epoch": best_epoch, "val_loss": best_val_loss}, f)
                        logger.info(f"  New best checkpoint (val_loss={val_loss:.6f}, step={best_step}, epoch={best_epoch})")
                dist.barrier()
            
            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    sample_fn = transport_sampler.sample_ode() # default to ode sampling
                    samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1] #Returns the state at all times. [-1] to take the final one
                    dist.barrier()

                    if use_cfg: #remove null samples
                        samples, _ = samples.chunk(2, dim=0)
                    out_samples = torch.zeros((args.global_batch_size, 4, args.image_size, args.image_size), device=device)
                    dist.all_gather_into_tensor(out_samples, samples)

                visuals_dir = os.path.join(experiment_dir, "visuals") if rank == 0 else None
                wandb_utils.log_image(out_samples, train_steps, visuals_dir)
                logging.info("Generating EMA samples done.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops-data-dir", type=str, required=True,
                        help="Directory containing ops_dataset.npz, ops_perturbation_map.json")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--class-dropout", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")
    parser.add_argument("--use-guidance", action="store_true", default=True,
                        help="Use guidance embedding module (default: True)")
    parser.add_argument("--use-direct-embed", action="store_true", default=False,
                        help="Use direct class mean embeddings as conditioning (no KL loss, no MLPs)")
    parser.add_argument("--embed-dim", type=int, default=384,
                        help="Dimension of encoder embeddings for guidance module")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight for KL loss term")
    parser.add_argument("--empirical-bayes-update-freq", type=int, default=10,
                        help="Update mu_eta/sigma_sq_eta every N steps (0 to disable)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of training data to use as validation set")
    parser.add_argument("--max-perturbations", type=int, default=None,
                        help="Number of perturbations to randomly select (from 1451). Default: all.")
    parser.add_argument("--imbalance-factor", type=float, default=1.0,
                        help="If < 1.0, drop (1-F) samples from one randomly chosen class.")

    parse_transport_args(parser)
    args = parser.parse_args()
    torch.serialization.add_safe_globals([argparse.Namespace])
    main(args)



# if rank == 0:
            #     raw_grid = make_channel_grid(x, nrow=4, normalize=True)
            #     Image.fromarray(raw_grid).save(os.path.join(experiment_dir, f"raw_samples_{rank}.png"))
            # with torch.no_grad():
            #     sample_model_kwargs['y'] = y
            #     sample_fn = transport_sampler.sample_ode() # default to ode sampling
            #     samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1] #Returns the state at all times. [-1] to take the final one
            #     dist.barrier()

            #     # if use_cfg: #remove null samples
            #     #     samples, _ = samples.chunk(2, dim=0)
            #     out_samples = torch.zeros((args.global_batch_size, 4, args.image_size, args.image_size), device=device)
            #     dist.all_gather_into_tensor(out_samples, samples)

            # visuals_dir = os.path.join(experiment_dir, "visuals") if rank == 0 else None
            # wandb_utils.log_image(out_samples, train_steps, visuals_dir)
            # print("Saved Image")