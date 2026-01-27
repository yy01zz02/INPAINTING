# Evaluation

Evaluation scripts for measuring inpainting quality metrics.

## Overview

This directory contains scripts for evaluating inpainting results against ground truth images using various quality metrics.

## Files

| File | Description |
|------|-------------|
| `client_metrics.py` | Core metrics calculator with server API clients and local metrics |
| `eval.py` | Main evaluation script for batch processing |

## Metrics

### Remote Metrics (via server API)
- **Image Reward** - Image quality reward score
- **HPS V2.1** - Human Preference Score v2.1
- **HPS v3** - Human Preference Score v3
- **Aesthetic Score** - CLIP-based aesthetic prediction
- **CLIP Score** - Text-image similarity
- **PickScore** - Preference-based scoring

### Local Metrics
- **PSNR** - Peak Signal-to-Noise Ratio
- **LPIPS** - Learned Perceptual Image Patch Similarity
- **MSE** - Mean Squared Error
- **Boundary Smoothness** - Gradient discontinuity at mask boundaries

## Usage

### Prerequisites

Start the reward model servers first (see `server/` directory):
```bash
cd ../server
python server_aesthetic.py &
python server_clip.py &
python server_hpsv2.py &
python server_pickscore.py &
```

### Run Evaluation

```bash
python eval.py --bench_dir /path/to/benchmark --model_name fill_grpo
```

### Data Format

Expected directory structure:
```
benchmark/
├── originals/      # Ground truth images
├── masks/          # Binary masks
├── prompts/        # Text prompts (.txt files)
└── results/
    └── model_name/ # Generated images
```

### Output

Results are saved to `benchmark/evaluations/`:
- `{bench}_{model}_summary.csv` - Averaged metrics
- `{bench}_{model}_detailed.csv` - Per-image metrics
