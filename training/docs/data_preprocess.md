# Data Preprocessing

This document describes data preprocessing for training inpainting models.

## Overview

To save GPU memory, we precompute text embeddings and VAE latents to eliminate the need to load the text encoder and VAE during training.

## Preprocessing Scripts

### Flux Fill Data

```bash
bash scripts/preprocess/preprocess_fill.sh
```

This will:
1. Load source images and masks
2. Compute T5 text embeddings
3. Compute VAE latents for images and masks
4. Save preprocessed data to the output directory

### Data Format

Input JSONL format:
```json
{
  "image_path": "path/to/image.jpg",
  "mask_path": "path/to/mask.png",
  "prompt": "description of the inpainting target"
}
```

Output structure:
```
output_dir/
├── prompt_embed/       # T5 text embeddings
├── pooled_prompt_embeds/  # Pooled embeddings
├── text_ids/           # Text token IDs
└── metadata.json       # Dataset index
```
