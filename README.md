# Inpainting with GRPO and DPO

## Overview

This repository contains the implementation of Group Relative Policy Optimization (GRPO) and Direct Preference Optimization (DPO) for training image inpainting diffusion models, with support for:

- **Flux.1 Fill** - Flow-based inpainting model
- **Stable Diffusion 1.5 Inpainting** - U-Net based inpainting model

## Repository Structure

```
.
├── dataset/            # Benchmark datasets samples (BrushBench, EditBench, FluxBench)
├── training/           # Core training framework
│   ├── fastvideo/      # Training scripts and modules
│   └── scripts/        # Launch scripts
├── inference/          # Inference scripts for Flux Fill and SD1.5
├── inference_brushnet/ # BrushNetX baseline inference
├── inference_powerpaint/ # PowerPaint baseline inference
├── evaluation/         # Evaluation scripts
├── reward_server/      # Reward model servers (main)
└── reward_server_extra/ # Additional reward model servers
```

## Models and Datasets

### 1. Benchmark Datasets

The `dataset/` directory contains sample data for the following benchmarks. Please refer to their official sources for full datasets:

*   **FluxBench**: [Official Website](https://bfl.ai/blog/24-11-21-tools)
*   **EditBench**: [Official Website](https://imagen.research.google/editor/)
*   **BrushBench**: [GitHub](https://github.com/TencentARC/BrushNet)

### 2. Setup Dependencies

Some evaluation and baseline models rely on third-party libraries. Please clone them into the respective directories:

```bash
# BrushNet (for inference_brushnet)
cd inference_brushnet
git clone https://github.com/TencentARC/BrushNet.git

# PowerPaint (for inference_powerpaint)
cd ../inference_powerpaint
git clone https://github.com/open-mmlab/PowerPaint.git

# HPSv3 (for reward_server_extra)
cd ../reward_server_extra/hpsv3
git clone https://github.com/MizzenAI/HPSv3.git
cd ../../
```

### 3. Inpainting Models

Please download the following models for training and inference:

*   **PrefPaint**: [GitHub](https://github.com/Kenkenzaii/PrefPaint)
*   **BrushNetX**: [HuggingFace](https://huggingface.co/TencentARC/BrushEdit/tree/main/brushnetX)
*   **PowerPaint v2**: [HuggingFace](https://huggingface.co/JunhaoZhuang/PowerPaint_v2)
*   **FLUX.1-Fill-dev**: [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
*   **Stable Diffusion Inpainting**: [HuggingFace](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)

### 3. Reward Models

The following reward models are used for evaluation and training (DPO/GRPO). 

| Model | Input | Output | Source | Note |
| :--- | :--- | :--- | :--- | :--- |
| **ImageReward** | Image + Text | Scalar | [HuggingFace](https://huggingface.co/THUDM/ImageReward) | - |
| **HPS v2.1** | Image + Text | Scalar | [GitHub](https://github.com/tgxs002/HPSv2) | - |
| **Aesthetic Score** | Image | Scalar | [GitHub](https://github.com/christophschuhmann/improved-aesthetic-predictor) | ViT-L-14 |
| **HPS v3** | Image + Text | Scalar | [GitHub](https://github.com/tgxs002/HPSv2) | Qwen2-VL-7B |
| **PickScore** | Image + Text | Scalar | [HuggingFace](https://huggingface.co/yuvalkirstain/PickScore_v1) | - |
| **CLIP** | Image + Text | Scalar | [HuggingFace](https://huggingface.co/openai/clip-vit-large-patch14) | ViT-L-14 |

For automated downloading and serving of these reward models, please refer to the `reward_server` and `reward_server_extra` directories.

## Quick Start

### 1. Setup Reward Servers

```bash
# Start reward model servers
cd reward_server
python server_aesthetic.py &   # Port 8161
python server_clip.py &        # Port 8162
python server_hpsv2.py &       # Port 8163
python server_pickscore.py &   # Port 8166

cd ../reward_server_extra
python hpsv3/server_hpsv3.py &      # Port 8164
python IR/server_imagereward.py &   # Port 8165
```

### 2. Training

```bash
cd training

# Flux Fill + GRPO
bash scripts/finetune/finetune_flux_fill_grpo_4gpus.sh

# SD1.5 + GRPO
bash scripts/finetune/finetune_sd_inpainting_grpo_4gpus.sh
```

### 3. Inference

```bash
cd inference

# Generate with Flux Fill models
python rlbench_fill.py --model_path /path/to/model
python rlbench_fill_grpo.py --model_path /path/to/model --finetuned_model_path /path/to/checkpoint

# Generate with SD1.5 models
python rlbench_sd15.py --model_path /path/to/model
python rlbench_sd15_grpo.py --model_path /path/to/model --unet_path /path/to/unet.safetensors
```

### 4. Evaluation

```bash
cd evaluation
python eval.py --bench_dir /path/to/benchmark --model_name fill_grpo
```

## Directory Details

| Directory | Purpose |
|-----------|---------|
| `training/` | Training framework with GRPO/DPO implementations |
| `inference/` | Inference scripts for our models (Flux Fill, SD1.5) |
| `inference_brushnet/` | BrushNetX baseline inference |
| `inference_powerpaint/` | PowerPaint v2 baseline inference |
| `evaluation/` | Evaluation metrics and scripts |
| `reward_server/` | Main reward model servers (Aesthetic, CLIP, HPSv2, PickScore) |
| `reward_server_extra/` | Additional servers (HPSv3, ImageReward) |

## Environment Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 12.4
- For Flux Fill: GPUs with >= 200GB VRAM
- For SD1.5: GPUs with >= 24GB VRAM

## License

Apache License 2.0
