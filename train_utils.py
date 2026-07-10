import os
from pathlib import Path
import json
import torch.distributed as dist
import logging
from time import time
import wandb_utils
from glob import glob

def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)

def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument("--diffusion-form", type=str, default="sigma", \
                        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],\
                        help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"],\
                        help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04, \
                        help="size of the last step taken")

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

def setup_logging_and_tracking(args):
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
    args.experiment_dir = experiment_dir
    args.checkpoint_dir = checkpoint_dir

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
    # if args.ckpt is not None:
    #     init_kwargs["rid"] = "97956759"
    #     init_kwargs["resume"] = "must"
    if args.wandb:
        wandb_utils.initialize(args, experiment_name, **init_kwargs)
    return args, logger