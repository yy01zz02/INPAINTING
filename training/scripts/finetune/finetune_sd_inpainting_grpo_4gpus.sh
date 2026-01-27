#!/bin/bash
# Training script for SD1.5 Inpainting model with GRPO
# Uses CLIP + HPS (1:1) API reward function
# Configuration: 4x96GB GPUs (no FSDP needed for SD1.5)

set -e

# Disable wandb and use swanlab
export WANDB_DISABLED=true


# Create necessary directories
mkdir -p images
mkdir -p data/outputs/sd15inpaint
mkdir -p logs



# NOTE: Set MODEL_PATH and DATA_JSON_PATH environment variables before running
# export MODEL_PATH="/path/to/stable-diffusion-inpainting"
# export DATA_JSON_PATH="/path/to/train_metadata.jsonl"

echo "Starting SD1.5 Inpainting GRPO training..."
echo "Configuration: 4 GPUs, No FSDP (SD1.5 fits in single GPU), CLIP + HPS API Reward (1:1)"

uv run accelerate launch --num_processes 4 --main_process_port 19005 fastvideo/train_grpo_sd_inpainting.py \
    --config=fastvideo/config_sd/inpainting.py \
    --config.seed=42 \
    --config.run_name="sd_inpainting_grpo" \
    --config.logdir="logs" \
    --config.num_epochs=10 \
    --config.save_freq=1 \
    --config.num_checkpoint_limit=5 \
    --config.mixed_precision="bf16" \
    --config.allow_tf32=True \
    --config.use_lora=False \
    --config.pretrained.model="${MODEL_PATH:-/path/to/stable-diffusion-inpainting}" \
    --config.sample.num_steps=50 \
    --config.sample.eta=1.0 \
    --config.sample.guidance_scale=7.5 \
    --config.sample.batch_size=4 \
    --config.train.batch_size=4 \
    --config.train.use_8bit_adam=False \
    --config.train.learning_rate=1e-5 \
    --config.train.gradient_accumulation_steps=4 \
    --config.train.max_grad_norm=1.0 \
    --config.train.num_inner_epochs=1 \
    --config.train.cfg=True \
    --config.train.adv_clip_max=5.0 \
    --config.train.clip_range=1e-4 \
    --config.data_json_path="${DATA_JSON_PATH:-/path/to/train_metadata.jsonl}" \
    --config.reward_type="clip" \
    --config.num_generations=8 \
    --config.max_train_steps=500 \
    --config.checkpoint_dir="data/outputs/sd15inpaint/checkpoints" \
    --config.final_model_dir="data/outputs/sd15inpaint/final_model"

echo "Training completed!"