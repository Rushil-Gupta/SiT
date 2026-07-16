import argparse
import torch
from train_utils import parse_transport_args

_emb_map = {
    "genept": 1536,
    "openphenom": 384,
    "cellprofiler": 312,
}

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops-data-dir", type=str, required=True,
                        help="Root dataset directory containing dataset subdirectories")
    parser.add_argument("--dataset", type=str, default="funk",
                        help="Dataset subdirectory within ops-data-dir (default: funk)")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--class-dropout", type=float, default=0.1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")
    parser.add_argument("--use-sample-embed", action="store_true", default=False,
                        help="Use sample embedding module (default: False)")
    parser.add_argument("--use-direct-embed", action="store_true", default=False,
                        help="Use direct class mean embeddings as conditioning (no KL loss, no MLPs)")
    parser.add_argument("--use-frozen-embed", action="store_true", default=False,
                        help="Use FrozenEmbeddingModule (precomputed embeddings + projection, no KL)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight for KL loss term (unused with use_frozen_embed)")
    parser.add_argument("--empirical-bayes-update-freq", type=int, default=10,
                        help="Update mu_eta/sigma_sq_eta every N steps (0 to disable)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of training data to use as validation set")
    parser.add_argument("--class-distribution-file", type=str, default=None,
                        help="Path to a JSON file describing the class-distribution scenario "
                             "(optional total_perturbations + tiers of imbalance factors). "
                             "See dataset/ops_dataset.py's OPSDataset docstring for the schema.")
    parser.add_argument("--cond-embedder", type=str, default="openphenom", choices=["openphenom", "genept", "cellprofiler"],
                        help="Embeddings for conditioning module (default: none, or 'genept' for GenePT embeddings)")
    parser.add_argument("--feature-extractor", type=str, default="openphenom",
                        choices=["openphenom", "cell_dino", "dinov2", "inception", "all"],
                        help="Feature extractor for evaluation metrics")
    parser.add_argument("--eval-samples-per-class", type=int, default=500,
                        help="Number of generated samples per class for metrics")
    parser.add_argument("--use-latent", action="store_true", default=False,
                        help="Train SiT in VAE latent space instead of pixel space")
    parser.add_argument("--latent-dir", type=str, default=None,
                        help="Directory with cached latents + latent_meta.json (see "
                             "dataset/funk/precompute_latents.py). Required if --use-latent.")
    parser.add_argument("--vae-checkpoint", type=str, default=None,
                        help="Path to finetuned VAE checkpoint (see finetune_vae.py). "
                             "Required for latent-space sampling.")

    parse_transport_args(parser)
    args = parser.parse_args()
    torch.serialization.add_safe_globals([argparse.Namespace])
    args.cond_dim = _emb_map[args.cond_embedder]
    return args