# RL Training Framework

Training framework for GRPO and DPO algorithms on image inpainting models.

## Overview

This repository implements Group Relative Policy Optimization (GRPO) and Direct Preference Optimization (DPO) for training image inpainting diffusion models. We support two backbone architectures:

- **Flux.1 Fill** - State-of-the-art flow-based inpainting model
- **Stable Diffusion 1.5 Inpainting** - Classic U-Net based inpainting model

## Directory Structure

```
training/
├── fastvideo/                    # Core training framework
│   ├── train_grpo_flux_fill.py   # Flux Fill + GRPO training
│   ├── train_dpo_flux_fill.py    # Flux Fill + DPO training  
│   ├── train_sft_flux_fill.py    # Flux Fill + SFT baseline
│   ├── train_grpo_sd_inpainting.py  # SD1.5 + GRPO training
│   ├── train_dpo_sd_inpainting.py   # SD1.5 + DPO training
│   ├── train_sft_sd_inpainting.py   # SD1.5 + SFT baseline
│   ├── config_sd/                # Configuration files for SD1.5
│   ├── dataset/                  # Dataset loaders
│   ├── models/                   # Model architectures
│   ├── rewards/                  # Reward functions
│   └── utils/                    # Utility functions
├── scripts/
│   └── finetune/                 # Training launch scripts
└── docs/                         # Documentation
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8
- 4x GPUs with >= 200GB VRAM (for Flux Fill)
- 4x GPUs with >= 24GB VRAM (for SD1.5)

## Installation

```bash
pip install -e .
# or using uv
uv pip install -e .
```

## Training

### Environment Setup

Set model paths via environment variables:

```bash
# For Flux Fill
export MODEL_PATH="/path/to/FLUX.1-Fill-dev"
export DATA_PATH="/path/to/preprocessed/data"

# For SD1.5 Inpainting
export MODEL_PATH="/path/to/stable-diffusion-inpainting"
export DATA_JSON_PATH="/path/to/train_metadata.jsonl"
```

### Launch Training

```bash
# Flux Fill + GRPO
bash scripts/finetune/finetune_flux_fill_grpo_4gpus.sh

# Flux Fill + DPO
bash scripts/finetune/finetune_flux_fill_dpo_4gpus.sh

# SD1.5 + GRPO
bash scripts/finetune/finetune_sd_inpainting_grpo_4gpus.sh

# SD1.5 + DPO
bash scripts/finetune/finetune_sd_inpainting_dpo_4gpus.sh
```

## Training Scripts

| Script | Model | Algorithm | Description |
|--------|-------|-----------|-------------|
| `train_grpo_flux_fill.py` | Flux Fill | GRPO | Full fine-tuning with FSDP, EMA |
| `train_dpo_flux_fill.py` | Flux Fill | Online DPO | With reference model updates |
| `train_sft_flux_fill.py` | Flux Fill | SFT | Supervised baseline |
| `train_grpo_sd_inpainting.py` | SD1.5 | GRPO | Multi-GPU with Accelerate |
| `train_dpo_sd_inpainting.py` | SD1.5 | Online DPO | With reference model updates |
| `train_sft_sd_inpainting.py` | SD1.5 | SFT | Supervised baseline |

## Key Features

- **GRPO Training**: Group-based relative policy optimization with clipped advantages
- **Online DPO**: Direct preference optimization with periodic reference updates
- **EMA Weights**: Exponential moving average for stable training
- **FSDP Support**: Fully sharded data parallel for large models (Flux Fill)
- **API-based Rewards**: Flexible reward computation via HTTP servers
- **Mixed Precision**: BF16/FP16 training support

## Data Format

Training data should be preprocessed into JSON format:

```json
{
    "image_path": "path/to/image.jpg",
    "mask_path": "path/to/mask.jpg", 
    "prompt": "description text",
    "t5_embedding_path": "path/to/embedding.pt"  // for Flux Fill
}
```

## License

Apache License 2.0
