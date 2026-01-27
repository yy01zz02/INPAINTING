# Inference-PowerPaint

Image generation scripts using PowerPaint v2 for baseline comparison.

## Overview

This directory contains inference scripts for generating inpainted images using PowerPaint v2 model on various benchmark datasets.

## Files

| File | Description |
|------|-------------|
| `rlbench_power.py` | PowerPaint v2 inference on RLBench dataset |
| `run_power.sh` | Batch script for running inference |
| `PowerPaint/` | PowerPaint library (submodule or local copy) |

## Environment Variables

Set model paths before running:

```bash
export POWERPAINT_PATH="/path/to/PowerPaint_v2"
export SD15_PATH="/path/to/stable-diffusion-v1-5"
export DATA_DIR="/path/to/rlbench"
```

## Usage

```bash
python rlbench_power.py \
    --checkpoint_dir $POWERPAINT_PATH \
    --sd15_path $SD15_PATH \
    --base_dir $DATA_DIR \
    --image_save_path ./results/powerpaint
```

## Data Format

Expected directory structure:
```
data/rlbench/
├── masks/           # Binary masks (white=inpaint region)
├── masked_images/   # Source images with mask applied
└── prompts/         # Text prompts (.txt files)
```

## Notes

- Uses UniPCMultistepScheduler for faster inference
- Default 50 DDIM steps with guidance scale 7.5
- Requires PowerPaint library in PowerPaint/ subdirectory
- Outputs saved as JPEG images
