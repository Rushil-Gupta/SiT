#!/bin/bash
# Finetunes FunkVAE (see vae.py) on 4-channel OPS microscopy images.
# Uses PyTorch DDP via torchrun (same pattern as train.sh) since this dataset
# is 1.1M+ images.

# Number of GPUs to use
NPROC_PER_NODE="gpu"

# Path to OPS data directory (contains ops_dataset.npz, ops_class_means.npy, ops_stats.json)
OPS_DATA_DIR="/scratch/rg5218/SiT/dataset"
DATASET="funk"

# Where finetuned VAE checkpoints get written
OUTPUT_DIR="/scratch/rg5218/SiT/vae_checkpoints"

# Training hyperparameters
EPOCHS=20
GLOBAL_BATCH_SIZE=128
LEARNING_RATE=4e-5
KL_WEIGHT=1e-5
VAL_SPLIT=0.05

# Logging and checkpointing
NUM_WORKERS=4
LOG_EVERY=50
CKPT_EVERY=1000
SEED=0

ARGS="
    --ops-data-dir $OPS_DATA_DIR
    --dataset $DATASET
    --output-dir $OUTPUT_DIR
    --epochs $EPOCHS
    --global-batch-size $GLOBAL_BATCH_SIZE
    --learning-rate $LEARNING_RATE
    --kl-weight $KL_WEIGHT
    --val-split $VAL_SPLIT
    --num-workers $NUM_WORKERS
    --log-every $LOG_EVERY
    --ckpt-every $CKPT_EVERY
    --seed $SEED
"

torchrun --nnodes=1 --nproc_per_node=$NPROC_PER_NODE --master_port=29501 finetune_vae.py $ARGS
