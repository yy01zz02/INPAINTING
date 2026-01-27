#!/bin/bash
# Generate images using BrushNetX on RLBench dataset
# Set environment variables before running

set -e

# Example usage:
# export BRUSHNET_PATH="/path/to/brushnetX"
# export BASE_MODEL_PATH="/path/to/realisticVisionV60B1_v51VAE"
# export DATA_DIR="/path/to/rlbench"
# bash run_brush.sh

python rlbench_brushnetx.py \
    --brushnet_path "${BRUSHNET_PATH:-/path/to/brushnetX}" \
    --base_model_path "${BASE_MODEL_PATH:-/path/to/base_model}" \
    --base_dir "${DATA_DIR:-./data/rlbench}" \
    --image_save_path "./results/brushnetx"
