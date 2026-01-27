#!/bin/bash
# Training script for Flux Fill inpainting model with GRPO
# Uses pre-computed T5 embeddings and CLIP + HPS (1:1) API reward function
# Configuration: 4x96GB GPUs with FSDP sharding

set -e

# Disable wandb and use swanlab
# export WANDB_DISABLED=true
# export SWANLAB_MODE=online

# Create necessary directories
mkdir -p images
mkdir -p data/outputs/fill_t_o



# NOTE: Set MODEL_PATH and DATA_PATH environment variables before running
# export MODEL_PATH="/path/to/FLUX.1-Fill-dev"
# export DATA_PATH="/path/to/preprocessed/data"

echo "Starting Flux Fill GRPO training..."
echo "Configuration: 4 GPUs, FSDP Full Shard, CLIP + HPS API Reward (1:1)"

uv run torchrun --nproc_per_node=4 --master_port 19004 \
    fastvideo/train_grpo_flux_fill.py \
    --seed 42 \
    --pretrained_model_name_or_path "${MODEL_PATH:-/path/to/FLUX.1-Fill-dev}" \
    --cache_dir data/.cache \
    --data_json_path data/fill_t/metadata.json \
    --gradient_checkpointing \
    --train_batch_size 2 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 150 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 25 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs/fill_t_o \
    --final_model_dir data/outputs/fill_t_final \
    --h 1024 \
    --w 1024 \
    --sampling_steps 20 \
    --eta 0.3 \
    --lr_warmup_steps 10 \
    --sampler_seed 1223627 \
    --max_grad_norm 0.1 \
    --weight_decay 0.0001 \
    --num_generations 4 \
    --shift 3.0 \
    --use_group \
    --timestep_fraction 0.6 \
    --clip_range 1e-3 \
    --adv_clip_max 5.0 \
    --use_ema \
    --ema_decay 0.995 \
    --init_same_noise \
    --fsdp_sharding_startegy full \
    --reward_type inpainting

echo "Training completed!"