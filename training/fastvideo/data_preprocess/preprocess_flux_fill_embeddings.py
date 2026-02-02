# Copyright (c) [2025] [FastVideo Team]
# SPDX-License-Identifier: [Apache License 2.0]

import argparse
import torch
import json
import os
import gc
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from diffusers import FluxPipeline
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
import numpy as np
from typing import List, Dict

class MetadataDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data_dir = os.path.dirname(jsonl_path)
        self.items = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        item['index'] = idx
        return item

def create_image_processors(vae_scale_factor: int = 8):
    """Create image processors matching official Flux pipeline.
    
    Flux uses vae_scale_factor * 2 for the image processor due to 2x2 patch packing.
    """
    # Image processor for regular images (normalize to [-1, 1])
    image_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor * 2,  # Flux uses 16 due to patch packing
        do_resize=True,
        do_normalize=True,  # Normalize to [-1, 1]
        do_convert_rgb=True,
    )
    
    # Mask processor for inpainting masks (no normalization, binarize)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor * 2,
        do_resize=True,
        do_normalize=False,  # Keep in [0, 1] range
        do_binarize=True,    # Binarize to 0/1
        do_convert_grayscale=True,
    )
    
    return image_processor, mask_processor


def load_image_with_processor(image_path: str, image_processor: VaeImageProcessor, height: int = 512, width: int = 512) -> torch.Tensor:
    """Load and preprocess image using VaeImageProcessor (matching official pipeline).
    
    Returns tensor with shape [1, C, H, W] in range [-1, 1].
    """
    image = Image.open(image_path).convert('RGB')
    # Preprocess returns [B, C, H, W] tensor normalized to [-1, 1]
    pixel_values = image_processor.preprocess(image, height=height, width=width)
    return pixel_values.float()


def load_mask_with_processor(mask_path: str, mask_processor: VaeImageProcessor, height: int = 512, width: int = 512) -> torch.Tensor:
    """Load and preprocess mask using VaeImageProcessor (matching official pipeline).
    
    Returns tensor with shape [1, 1, H, W] in range [0, 1], binarized.
    """
    mask = Image.open(mask_path).convert('L')
    # Preprocess returns [B, 1, H, W] tensor in [0, 1] range, binarized
    mask_values = mask_processor.preprocess(mask, height=height, width=width)
    return mask_values.float()

def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    subdirs = ["prompt_embed", "pooled_prompt_embeds", "text_ids", "source_latents", "mask_latents", "masked_latents"]
        
    for d in subdirs:
        os.makedirs(os.path.join(args.output_dir, d), exist_ok=True)

    # Load Model Components
    # We load the full pipeline to ensure we get the correct tokenizer/encoders paired
    # But we can move the transformer to CPU or not load it if possible.
    # For simplicity with Flux, loading the pipeline is often easiest, but we can offload.
    
    print(f"Rank {local_rank}: Loading Pipeline...")
    pipe = FluxPipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(device)
    
    # We don't need the transformer for preprocessing
    pipe.transformer = None
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create image processors using VAE scale factor from pipeline
    vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)  # Usually 8
    image_processor, mask_processor = create_image_processors(vae_scale_factor)
    print(f"Rank {local_rank}: Using VAE scale factor {vae_scale_factor}, image_processor scale factor {vae_scale_factor * 2}")

    dataset = MetadataDataset(args.jsonl_path)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        sampler=sampler, 
        num_workers=args.dataloader_num_workers,
        collate_fn=lambda x: x[0] # Return single item directly
    )

    processed_items = []

    print(f"Rank {local_rank}: Starting processing...")
    for item in tqdm(dataloader, disable=local_rank != 0):
        try:
            filename_base = f"fill_{item['index']}"
            
            # 1. Process Text
            prompt = item['prompt'] # Edit prompt or Inpainting prompt
            
            with torch.no_grad():
                # Flux encode_prompt returns: prompt_embeds, pooled_prompt_embeds, text_ids
                prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=prompt, 
                    prompt_2=prompt
                )
            
            # Save Text Embeddings
            # prompt_embeds shape: (batch, seq_len, hidden_dim) -> save (seq_len, hidden_dim)
            # pooled_prompt_embeds shape: (batch, hidden_dim) -> save (hidden_dim,)
            # text_ids shape: (seq_len, 3) -> save full tensor (NOT text_ids[0] which would only save first row)
            torch.save(prompt_embeds[0].cpu(), os.path.join(args.output_dir, "prompt_embed", f"{filename_base}.pt"))
            torch.save(pooled_prompt_embeds[0].cpu(), os.path.join(args.output_dir, "pooled_prompt_embeds", f"{filename_base}.pt"))
            torch.save(text_ids.cpu(), os.path.join(args.output_dir, "text_ids", f"{filename_base}.pt"))
            
            new_item = {
                "caption": prompt,
                "prompt_embed_path": f"{filename_base}.pt",
                "pooled_prompt_embeds_path": f"{filename_base}.pt",
                "text_ids": f"{filename_base}.pt",
            }

            # 2. Process Images for Fill/Inpainting Task
            # Load Image and Mask using official VaeImageProcessor
            image_path = os.path.join(dataset.data_dir, item['image'])
            mask_path = os.path.join(dataset.data_dir, item['mask'])
                
            # Use VaeImageProcessor for image (normalized to [-1, 1])
            pixel_values = load_image_with_processor(image_path, image_processor, args.height, args.width).to(device)
            # Use mask_processor for mask (in [0, 1] range, binarized)
            mask_values = load_mask_with_processor(mask_path, mask_processor, args.height, args.width).to(device)
            
            # Mask is already binarized by mask_processor, create masked image
            # Note: pixel_values is in [-1, 1], mask_values is in [0, 1]
            # masked_image should keep original where mask=0, and set to neutral where mask=1
            masked_image = pixel_values * (1 - mask_values)
            
            with torch.no_grad():
                # Convert to bfloat16 right before VAE encoding to match VAE weights
                pixel_values_bf16 = pixel_values.to(dtype=torch.bfloat16)
                masked_image_bf16 = masked_image.to(dtype=torch.bfloat16)
                
                # GT Latents (ground truth for loss calculation)
                source_latents = pipe.vae.encode(pixel_values_bf16).latent_dist.sample()
                source_latents = (source_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                
                # Masked Latents (masked image encoded)
                masked_latents = pipe.vae.encode(masked_image_bf16).latent_dist.sample()
                masked_latents = (masked_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                
                # Save mask at original pixel size for official reshape in training
                # Official FluxFillPipeline expects mask at pixel resolution [1, H, W]
                # Training script will reshape [H, 8, W, 8] -> [64, H/8, W/8] to preserve edge details
                # mask_values is already [1, 1, H, W], save as [1, H, W]
                mask_pixel = mask_values.squeeze(0).to(dtype=torch.bfloat16)  # [1, H, W]

            torch.save(source_latents[0].cpu(), os.path.join(args.output_dir, "source_latents", f"{filename_base}.pt"))
            torch.save(masked_latents[0].cpu(), os.path.join(args.output_dir, "masked_latents", f"{filename_base}_masked.pt"))
            torch.save(mask_pixel.cpu(), os.path.join(args.output_dir, "mask_latents", f"{filename_base}_mask.pt"))
            
            new_item["source_latents_path"] = f"{filename_base}.pt"
            new_item["masked_latents_path"] = f"{filename_base}_masked.pt"
            new_item["mask_latents_path"] = f"{filename_base}_mask.pt"
            # Save original mask path for reward computation
            new_item["mask_path"] = item['mask']

            processed_items.append(new_item)

        except Exception as e:
            print(f"Rank {local_rank} Error processing {item['index']}: {e}")

    # Gather results
    all_processed_items = [None] * world_size
    dist.all_gather_object(all_processed_items, processed_items)
    
    if local_rank == 0:
        flat_items = [item for sublist in all_processed_items for item in sublist]
        output_json = os.path.join(args.output_dir, "metadata.json")
        with open(output_json, 'w') as f:
            json.dump(flat_items, f, indent=4)
        print(f"Saved metadata to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=None, help="If set, use same value for both height and width (square)")
    parser.add_argument("--height", type=int, default=512, help="Image height (must be divisible by 16)")
    parser.add_argument("--width", type=int, default=512, help="Image width (must be divisible by 16)")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # If image_size is provided, use it for both height and width (backward compatibility)
    if args.image_size is not None:
        args.height = args.image_size
        args.width = args.image_size
    
    main(args)