# Inference-BrushNet

Image generation scripts using BrushNetX (BrushEdit) for baseline comparison.

## Overview

This directory contains inference scripts for generating inpainted images using BrushNetX model on various benchmark datasets.

## Files

| File | Description |
|------|-------------|
| `rlbench_brushnetx.py` | BrushNetX inference on RLBench dataset |
| `run_brush.sh` | Batch script for running inference |
| `BrushNet/` | BrushNet library (submodule or local copy) |

## Environment Variables

Set model paths before running:

```bash
export BRUSHNET_PATH="/path/to/brushnetX"
export BASE_MODEL_PATH="/path/to/realisticVisionV60B1_v51VAE"
export DATA_DIR="/path/to/rlbench"
```

## Usage

```bash
python rlbench_brushnetx.py \
    --brushnet_path $BRUSHNET_PATH \
    --base_model_path $BASE_MODEL_PATH \
    --base_dir $DATA_DIR \
    --image_save_path ./results/brushnetx
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
- Default 50 inference steps
- Outputs saved as JPEG images
