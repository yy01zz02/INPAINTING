# Inference

Image generation scripts for evaluating inpainting models on RLBench benchmark.

## Overview

This directory contains scripts for generating inpainted images using different models and fine-tuning methods.

## Files

| File | Description |
|------|-------------|
| `rlbench_fill.py` | Flux.1 Fill baseline inference |
| `rlbench_fill_grpo.py` | Flux.1 Fill + GRPO fine-tuned inference |
| `rlbench_fill_sft.py` | Flux.1 Fill + SFT fine-tuned inference |
| `rlbench_sd15.py` | SD1.5 Inpainting baseline inference |
| `rlbench_sd15_grpo.py` | SD1.5 Inpainting + GRPO fine-tuned inference |
| `rlbench_sd15_sft.py` | SD1.5 Inpainting + SFT fine-tuned inference |
| `rlbench_prefpaint.py` | PrefPaint baseline inference |
| `run_flux.sh` | Batch script for Flux.1 Fill models |
| `run_sd15.sh` | Batch script for SD1.5 Inpainting models |
| `run_prefpaint.sh` | Batch script for PrefPaint baseline |

## Usage

### Environment Variables

Set the following environment variables before running:

```bash
# For Flux.1 Fill
export MODEL_PATH="/path/to/FLUX.1-Fill-dev"

# For SD1.5 Inpainting
export MODEL_PATH="/path/to/stable-diffusion-inpainting"

# Data directory (containing masks/, masked_images/, prompts/)
export DATA_DIR="/path/to/rlbench"
```

### Running Generation

```bash
# Flux.1 Fill baseline
python rlbench_fill.py --model_path $MODEL_PATH --base_dir $DATA_DIR

# Flux.1 Fill + GRPO
python rlbench_fill_grpo.py --model_path $MODEL_PATH --finetuned_model_path /path/to/checkpoint

# SD1.5 baseline
python rlbench_sd15.py --model_path $MODEL_PATH --base_dir $DATA_DIR

# SD1.5 + GRPO
python rlbench_sd15_grpo.py --model_path $MODEL_PATH --unet_path /path/to/unet.safetensors

# PrefPaint baseline
export PREFPAINT_MODEL_PATH="/path/to/prefpaint"
export RLBENCH_DIR="/path/to/rlbench"
python rlbench_prefpaint.py
```

## Data Format

Expected directory structure:
```
data/rlbench/
├── masks/           # Binary masks (white=inpaint region)
├── masked_images/   # Source images with mask applied
└── prompts/         # Text prompts (.txt files)
```
