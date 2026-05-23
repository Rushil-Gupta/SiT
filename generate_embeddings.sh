#!/bin/bash
# Generate OPS embeddings with the new preprocessing (arcsinh + Normalize(7,7))

DATA_DIR=${1:-"/path/to/ops_data"}
MODEL_NAME=${2:-"recursionpharma/OpenPhenom"}
BATCH_SIZE=${3:-256}
NUM_GPUS=${4:-"all"}

python generate_embeddings.py \
    --data-dir "$DATA_DIR" \
    --model-name "$MODEL_NAME" \
    --batch-size "$BATCH_SIZE" \
    --num-gpus "$NUM_GPUS"
