#!/bin/bash
# Generate images using SD1.5 Inpainting models on RLBench dataset
# Set MODEL_PATH and DATA_DIR environment variables before running

set -e

# Example usage:
# export MODEL_PATH="/path/to/stable-diffusion-inpainting"
# export DATA_DIR="/path/to/rlbench"
# bash run_sd15.sh

# Baseline
python rlbench_sd15.py \
    --model_path "${MODEL_PATH:-/path/to/stable-diffusion-inpainting}" \
    --base_dir "${DATA_DIR:-./data/rlbench}" \
    --image_save_path "./results/sd15"

# GRPO
python rlbench_sd15_grpo.py \
    --model_path "${MODEL_PATH:-/path/to/stable-diffusion-inpainting}" \
    --unet_path "${GRPO_UNET:-./checkpoints/grpo/diffusion_pytorch_model.safetensors}" \
    --base_dir "${DATA_DIR:-./data/rlbench}" \
    --image_save_path "./results/sd15_grpo"

# SFT
python rlbench_sd15_sft.py \
    --model_path "${MODEL_PATH:-/path/to/stable-diffusion-inpainting}" \
    --unet_path "${SFT_UNET:-./checkpoints/sft/diffusion_pytorch_model.safetensors}" \
    --base_dir "${DATA_DIR:-./data/rlbench}" \
    --image_save_path "./results/sd15_sft"
