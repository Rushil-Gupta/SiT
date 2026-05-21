#!/bin/bash
# Training script for CellDiffusion (SiT with guidance embeddings)

# Number of GPUs to use
NPROC_PER_NODE=4

# Path to OPS data directory (contains ops_dataset.npz, ops_class_means.npy, ops_stats.json)
OPS_DATA_DIR="/scratch/rg5218/SiT"

# Model variant: SiT-S/2, SiT-S/4, SiT-B/2, SiT-B/4, SiT-L/2, SiT-XL/2
MODEL="SiT-S/2"

# Training hyperparameters
IMAGE_SIZE=100
GLOBAL_BATCH_SIZE=128
EPOCHS=1400
LEARNING_RATE=1e-4

# Loss hyperparameters
BETA=1.0              # KL weight
CFG_SCALE=4.0         # Classifier-free guidance scale
CLASS_DROPOUT=0.1     # Label dropout probability for CFG

# Transport settings
PATH_TYPE="Linear"    # Linear, GVP, VP
PREDICTION="velocity" # velocity, score, noise
LOSS_WEIGHT="None"    # None, velocity, likelihood

# Logging and checkpointing
LOG_EVERY=100
CKPT_EVERY=50000
SAMPLE_EVERY=10000
RESULTS_DIR="results"

# WandB (set ENTITY and PROJECT env vars, then add --wandb flag)
export ENTITY="rg5218-new-york-university"
export PROJECT="CellDiffusion"
export WANDB_KEY="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"

# Run training
torchrun --nnodes=1 --nproc_per_node=$NPROC_PER_NODE train.py \
    --model $MODEL \
    --ops-data-dir $OPS_DATA_DIR \
    --image-size $IMAGE_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --epochs $EPOCHS \
    --beta $BETA \
    --cfg-scale $CFG_SCALE \
    --path-type $PATH_TYPE \
    --prediction $PREDICTION \
    --loss-weight $LOSS_WEIGHT \
    --log-every $LOG_EVERY \
    --ckpt-every $CKPT_EVERY \
    --sample-every $SAMPLE_EVERY \
    --results-dir $RESULTS_DIR \
    --num-workers 4 \
    --global-seed 0 \
    --embed-dim 384 \
    --use-guidance
