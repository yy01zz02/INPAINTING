#!/bin/bash
# Training script for SD1.5 Inpainting model with SFT (Full Fine-tuning)
# Supervised Fine-Tuning for inpainting task
# Configuration: 4 GPUs (no FSDP needed for SD1.5)

set -e

# Disable wandb and use swanlab
export WANDB_DISABLED=true

# Create necessary directories
mkdir -p images
mkdir -p data/outputs/sd15inpaint_sft
mkdir -p logs

echo "Starting SD1.5 Inpainting SFT training (Full Fine-tuning)..."
echo "Configuration: 4 GPUs, No FSDP (SD1.5 fits in single GPU)"

# Run with accelerate for multi-GPU support
uv run accelerate launch --num_processes 4 --main_process_port 19007 fastvideo/train_sft_sd_inpainting.py \
    --config=fastvideo/config_sd/sft.py \
    --config.seed=42 \
    --config.run_name="sd_inpainting_sft" \
    --config.logdir="logs" \
    --config.num_epochs=10 \
    --config.save_freq=100 \
    --config.num_checkpoint_limit=5 \
    --config.mixed_precision="bf16" \
    --config.allow_tf32=True \
    --config.pretrained.model="<PATH_TO_SD15_INPAINTING_MODEL>" \
    --config.train.batch_size=2 \
    --config.train.use_8bit_adam=False \
    --config.train.learning_rate=1e-5 \
    --config.train.gradient_accumulation_steps=4 \
    --config.train.max_grad_norm=1.0 \
    --config.train.cfg=False \
    --config.data_json_path="data/inpainting/train_metadata.jsonl" \
    --config.max_train_steps=1000 \
    --config.checkpoint_dir="data/outputs/sd15inpaint_sft/checkpoints" \
    --config.final_model_dir="data/outputs/sd15inpaint_sft/final_model"

echo "Training completed!"
