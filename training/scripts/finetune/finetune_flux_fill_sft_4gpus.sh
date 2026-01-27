#!/bin/bash
# Training script for Flux Fill inpainting model with SFT (Full Fine-tuning)
# Supervised Fine-Tuning for inpainting task
# Configuration: 4 GPUs with FSDP sharding

set -e

# Disable wandb and use swanlab
export WANDB_DISABLED=true

# Create necessary directories
mkdir -p images
mkdir -p data/outputs/flux_fill_sft
mkdir -p logs

# NOTE: Set MODEL_PATH and DATA_JSON_PATH environment variables before running
# export MODEL_PATH="/path/to/FLUX.1-Fill-dev"
# export DATA_JSON_PATH="/path/to/train_preprocessed.json"

echo "Starting Flux Fill SFT training (Full Fine-tuning)..."
echo "Configuration: 4 GPUs with FSDP sharding"

uv run torchrun --nproc_per_node=4 --master_port=19008 fastvideo/train_sft_flux_fill.py \
    --model_path="${MODEL_PATH:-/path/to/FLUX.1-Fill-dev}" \
    --data_json_path="${DATA_JSON_PATH:-/path/to/train_preprocessed.json}" \
    --output_dir="data/outputs/flux_fill_sft" \
    --final_model_dir="data/outputs/flux_fill_sft/final_model" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-5 \
    --weight_decay=1e-4 \
    --max_grad_norm=1.0 \
    --max_train_steps=1000 \
    --num_epochs=100 \
    --warmup_steps=100 \
    --lr_scheduler="cosine" \
    --h=512 \
    --w=512 \
    --sampling_steps=28 \
    --shift=1.0 \
    --fsdp_sharding_strategy="full_shard" \
    --gradient_checkpointing \
    --sp_size=1 \
    --seed=42 \
    --allow_tf32 \
    --save_steps=100 \
    --num_workers=4

echo "Training completed!"
