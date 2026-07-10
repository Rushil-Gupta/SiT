# #!/bin/bash
# # Training script for CellDiffusion (SiT with guidance embeddings)

# # Number of GPUs to use
# NPROC_PER_NODE="gpu"

# # Path to OPS data directory (contains ops_dataset.npz, ops_class_means.npy, ops_stats.json)
# OPS_DATA_DIR="/scratch/rg5218/SiT/dataset"

# # Model variant: SiT-S/2, SiT-S/4, SiT-B/2, SiT-B/4, SiT-L/2, SiT-XL/2
# MODEL="SiT-S/2"

# # Training hyperparameters
# IMAGE_SIZE=100
# GLOBAL_BATCH_SIZE=128
# EPOCHS=100 #1400 before, but we can start with 100 for testing and then scale up
# LEARNING_RATE=1e-5
# VAL_SPLIT=0.01
# MAX_PERTURBATIONS=12  # Set to a number (e.g. 50) to subset perturbations; empty = all 1451
# IMBALANCE_FACTOR=0.01  # If < 1.0, drop (1-F) samples from one randomly chosen class

# # Loss hyperparameters
# BETA=0.1              # KL weight
# EB_UPDATE_FREQ=1000       # Empirical Bayes prior update frequency (steps)
# CFG_SCALE=1.0         # Classifier-free guidance scale
# CLASS_DROPOUT=0.2     # Label dropout probability for CFG

# # Transport settings
# PATH_TYPE="Linear"    # Linear, GVP, VP
# PREDICTION="velocity" # velocity, score, noise
# LOSS_WEIGHT="None"    # None, velocity, likelihood

# # Logging and checkpointing
# LOG_EVERY=10
# CKPT_EVERY=100 #50k before, but we can start with 5k for testing and then scale up
# SAMPLE_EVERY=50 # 10k before, but we can start with 1k for testing and then scale up
# RESULTS_DIR="results"

# # CKPT="/home/rg5218/scratch/SiT/results/008-SiT-S-2-Linear-velocity-None/checkpoints/0010000.pt"

# # WandB (set ENTITY and PROJECT env vars, then add --wandb flag)
# export ENTITY="rg5218-new-york-university"
# export PROJECT="CellDiffusion"
# export WANDB_KEY="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"

# # Build args (conditionally include max-perturbations)
# ARGS="
#     --model $MODEL
#     --ops-data-dir $OPS_DATA_DIR
#     --dataset funk
#     --image-size $IMAGE_SIZE
#     --global-batch-size $GLOBAL_BATCH_SIZE
#     --epochs $EPOCHS
#     --beta $BETA
#     --empirical-bayes-update-freq $EB_UPDATE_FREQ
#     --cfg-scale $CFG_SCALE
#     --class-dropout $CLASS_DROPOUT
#     --path-type $PATH_TYPE
#     --prediction $PREDICTION
#     --loss-weight $LOSS_WEIGHT
#     --log-every $LOG_EVERY
#     --ckpt-every $CKPT_EVERY
#     --sample-every $SAMPLE_EVERY
#     --results-dir $RESULTS_DIR
#     --num-workers 4
#     --global-seed 0
#     --embed-dim 384
#     --use-guidance
#     --wandb
#     --val-split $VAL_SPLIT
#     --imbalance-factor $IMBALANCE_FACTOR
# "
# if [ -n "$MAX_PERTURBATIONS" ]; then
#     ARGS="$ARGS --max-perturbations $MAX_PERTURBATIONS"
# fi

# # Run training
# torchrun --nnodes=1 --nproc_per_node=$NPROC_PER_NODE train.py $ARGS

#!/bin/bash
# Training script for CellDiffusion (SiT with guidance embeddings)

# Number of GPUs to use
NPROC_PER_NODE="gpu"

# Path to OPS data directory (contains ops_dataset.npz, ops_class_means.npy, ops_stats.json)
OPS_DATA_DIR="/scratch/rg5218/SiT/dataset"

# Model variant: SiT-S/2, SiT-S/4, SiT-B/2, SiT-B/4, SiT-L/2, SiT-XL/2
MODEL="SiT-B/2"

# Training hyperparameters
IMAGE_SIZE=100
GLOBAL_BATCH_SIZE=256
EPOCHS=700 #1400 before, but we can start with 100 for testing and then scale up
LEARNING_RATE=1e-5
VAL_SPLIT=0.01
MAX_PERTURBATIONS=12  # Set to a number (e.g. 50) to subset perturbations; empty = all 1451
IMBALANCE_FACTOR=0.01  # If < 1.0, drop (1-F) samples from one randomly chosen class

# Loss hyperparameters
BETA=0.1              # KL weight
EB_UPDATE_FREQ=1000       # Empirical Bayes prior update frequency (steps)
CFG_SCALE=6.0         # Classifier-free guidance scale
CLASS_DROPOUT=0.2     # Label dropout probability for CFG

# Transport settings
PATH_TYPE="Linear"    # Linear, GVP, VP
PREDICTION="velocity" # velocity, score, noise
LOSS_WEIGHT="None"    # None, velocity, likelihood

# Logging and checkpointing
LOG_EVERY=50
CKPT_EVERY=1000 #50k before, but we can start with 5k for testing and then scale up
SAMPLE_EVERY=1000 # 10k before, but we can start with 1k for testing and then scale up
RESULTS_DIR="results"

CKPT=""

# WandB (set ENTITY and PROJECT env vars, then add --wandb flag)
export ENTITY="rg5218-new-york-university"
export PROJECT="CellDiffusion"
export WANDB_KEY="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"

# Build args (conditionally include max-perturbations)
ARGS="
    --model $MODEL
    --ops-data-dir $OPS_DATA_DIR
    --dataset funk
    --image-size $IMAGE_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --epochs $EPOCHS
    --beta $BETA
    --empirical-bayes-update-freq $EB_UPDATE_FREQ
    --cfg-scale $CFG_SCALE
    --class-dropout $CLASS_DROPOUT
    --path-type $PATH_TYPE
    --prediction $PREDICTION
    --loss-weight $LOSS_WEIGHT
    --log-every $LOG_EVERY
    --ckpt-every $CKPT_EVERY
    --sample-every $SAMPLE_EVERY
    --results-dir $RESULTS_DIR
    --num-workers 4
    --global-seed 0
    --embed-dim 384
    --wandb
    --val-split $VAL_SPLIT
    --imbalance-factor $IMBALANCE_FACTOR
    --use-frozen-embed
"
if [ -n "$MAX_PERTURBATIONS" ]; then
    ARGS="$ARGS --max-perturbations $MAX_PERTURBATIONS"
fi

export CUDA_HOME=/usr/local/cuda-12.6
export CPATH=$CUDA_HOME/include:$CPATH

# Run training
torchrun --nnodes=1 --nproc_per_node=$NPROC_PER_NODE train.py $ARGS

