# Dataset for Flux Fill inpainting task.

import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
from PIL import Image


class FluxFillLatentDataset(Dataset):
    """Dataset with pre-computed latents and T5 embeddings for Flux Fill inpainting.
    
    This dataset expects pre-processed embeddings stored in subdirectories.
    """
    
    def __init__(
        self,
        json_path: str,
        num_latent_t: int = 1,
        cfg_rate: float = 0.0,
        image_size: int = 512,
    ):
        self.json_path = json_path
        self.cfg_rate = cfg_rate
        self.data_dir = os.path.dirname(json_path)
        self.image_size = image_size
        
        # Directories for pre-computed embeddings
        self.prompt_embed_dir = os.path.join(self.data_dir, "prompt_embed")
        self.pooled_prompt_embeds_dir = os.path.join(self.data_dir, "pooled_prompt_embeds")
        self.text_ids_dir = os.path.join(self.data_dir, "text_ids")
        self.masked_latents_dir = os.path.join(self.data_dir, "masked_latents")
        self.mask_latents_dir = os.path.join(self.data_dir, "mask_latents")
        
        # Load data annotations
        with open(json_path, 'r') as f:
            self.data_anno = json.load(f)
        
        self.num_latent_t = num_latent_t
        
        self.lengths = [
            data_item.get("length", 1) for data_item in self.data_anno
        ]
    
    def __len__(self):
        return len(self.data_anno)
    
    def _load_mask_pil(self, mask_path: str) -> Image.Image:
        """Load original mask as PIL Image for reward computation."""
        full_path = os.path.join(self.data_dir, mask_path)
        if os.path.exists(full_path):
            mask = Image.open(full_path).convert('L')
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            return mask
        else:
            # If mask path doesn't exist, return a default gray mask
            return Image.new('L', (self.image_size, self.image_size), 128)
    
    def __getitem__(self, idx):
        item = self.data_anno[idx]
        
        # Load prompt embeddings (T5)
        prompt_embed_file = item["prompt_embed_path"]
        pooled_prompt_embeds_file = item["pooled_prompt_embeds_path"]
        text_ids_file = item["text_ids"]
        masked_latents_file = item["masked_latents_path"]
        mask_latents_file = item["mask_latents_path"]
        
        prompt_embed = torch.load(
            os.path.join(self.prompt_embed_dir, prompt_embed_file),
            map_location="cpu",
            weights_only=True,
        )
        
        pooled_prompt_embeds = torch.load(
            os.path.join(self.pooled_prompt_embeds_dir, pooled_prompt_embeds_file),
            map_location="cpu",
            weights_only=True,
        )
        
        text_ids = torch.load(
            os.path.join(self.text_ids_dir, text_ids_file),
            map_location="cpu",
            weights_only=True,
        )
        
        # Load masked image latents (source * (1-mask))
        masked_latents = torch.load(
            os.path.join(self.masked_latents_dir, masked_latents_file),
            map_location="cpu",
            weights_only=True,
        )
        
        # Load mask at pixel resolution [1, H, W] for official reshape in training
        # Training script will reshape [H, 8, W, 8] -> [64, H/8, W/8] to preserve edge details
        mask_latents = torch.load(
            os.path.join(self.mask_latents_dir, mask_latents_file),
            map_location="cpu",
            weights_only=True,
        )
        
        caption = item['caption']
        
        # Load original mask PIL for reward computation (if available)
        mask_path = item.get('mask_path', None)
        if mask_path:
            mask_pil = self._load_mask_pil(mask_path)
        else:
            mask_pil = None
        
        return (
            prompt_embed,
            pooled_prompt_embeds,
            text_ids,
            masked_latents,
            mask_latents,
            caption,
            mask_pil,
        )


def flux_fill_latent_collate_function(batch):
    """Collate function for pre-computed latent dataset."""
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
        masked_latents,
        mask_latents,
        captions,
        mask_pils,
    ) = zip(*batch)
    
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    pooled_prompt_embeds = torch.stack(pooled_prompt_embeds, dim=0)
    text_ids = torch.stack(text_ids, dim=0)
    masked_latents = torch.stack(masked_latents, dim=0)
    mask_latents = torch.stack(mask_latents, dim=0)
    
    return (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
        masked_latents,
        mask_latents,
        captions,
        mask_pils,
    )