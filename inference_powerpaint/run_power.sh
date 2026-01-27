#!/bin/bash
# Generate images using PowerPaint v2 on RLBench dataset
# Set environment variables before running

set -e

# Example usage:
# export POWERPAINT_PATH="/path/to/PowerPaint_v2"
# export SD15_PATH="/path/to/stable-diffusion-v1-5"
# export DATA_DIR="/path/to/rlbench"
# bash run_power.sh

python rlbench_power.py \
    --checkpoint_dir "${POWERPAINT_PATH:-/path/to/PowerPaint_v2}" \
    --sd15_path "${SD15_PATH:-/path/to/stable-diffusion-v1-5}" \
    --base_dir "${DATA_DIR:-./data/rlbench}" \
    --image_save_path "./results/powerpaint"
