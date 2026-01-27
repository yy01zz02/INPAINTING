#!/bin/bash
# Generate images using Flux.1 Fill models on RLBench dataset
# Set MODEL_PATH and DATA_DIR environment variables before running

set -e

# Example usage:
# export MODEL_PATH="/path/to/FLUX.1-Fill-dev"
# export DATA_DIR="/path/to/rlbench"
# bash run_flux.sh

# Baseline
python rlbench_fill.py \
    --model_path "${MODEL_PATH:-/path/to/FLUX.1-Fill-dev}" \
    --base_dir "${DATA_DIR:-./data/rlbench}" \
    --image_save_path "./results/fill"

# GRPO
python rlbench_fill_grpo.py \
    --model_path "${MODEL_PATH:-/path/to/FLUX.1-Fill-dev}" \
    --finetuned_model_path "${GRPO_CHECKPOINT:-./checkpoints/grpo}" \
    --base_dir "${DATA_DIR:-./data/rlbench}" \
    --image_save_path "./results/fill_grpo"

# SFT
python rlbench_fill_sft.py \
    --model_path "${MODEL_PATH:-/path/to/FLUX.1-Fill-dev}" \
    --finetuned_model_path "${SFT_CHECKPOINT:-./checkpoints/sft}" \
    --base_dir "${DATA_DIR:-./data/rlbench}" \
    --image_save_path "./results/fill_sft"
