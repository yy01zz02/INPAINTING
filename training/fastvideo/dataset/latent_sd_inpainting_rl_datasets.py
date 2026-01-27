# Dataset for SD1.5 Inpainting task.
# Data format: {"prompt": "...", "image": "...", "mask": "..."}

import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import numpy as np
from torchvision import transforms


class SDInpaintingDataset(Dataset):
    """Dataset for Stable Diffusion 1.5 Inpainting task.
    
    Expected JSONL format:
    {
        "prompt": "Calle De Portugal Acuarela S Papel 55x36 Cm Watercolor Aquarelle",
        "image": "images/train_000000.png",
        "mask": "masks/train_000000.png"
    }
    """
    
    def __init__(
        self,
        jsonl_path: str,
        image_size: int = 512,
        cfg_rate: float = 0.0,
    ):
        self.jsonl_path = jsonl_path
        self.image_size = image_size
        self.cfg_rate = cfg_rate
        self.data_dir = os.path.dirname(jsonl_path)
        
        # Load data from JSONL
        self.data_anno = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data_anno.append(json.loads(line))
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        
        self.lengths = [1 for _ in self.data_anno]
    
    def __len__(self):
        return len(self.data_anno)
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load image."""
        full_path = os.path.join(self.data_dir, image_path)
        image = Image.open(full_path).convert('RGB')
        return image
    
    def _load_mask(self, mask_path: str) -> Image.Image:
        """Load mask."""
        full_path = os.path.join(self.data_dir, mask_path)
        mask = Image.open(full_path).convert('L')  # Grayscale
        return mask
    
    def __getitem__(self, idx):
        item = self.data_anno[idx]
        
        prompt = item['prompt']
        image_path = item['image']
        mask_path = item['mask']
        
        # Load images
        source_image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        
        # Resize for pipeline (keep PIL format for SD pipeline compatibility)
        source_image_resized = source_image.resize((self.image_size, self.image_size))
        mask_resized = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        # Apply transforms for tensor versions
        image_tensor = self.image_transform(source_image)
        mask_tensor = self.mask_transform(mask)
        
        # Binarize mask (0 or 1)
        mask_tensor = (mask_tensor > 0.5).float()
        
        # Create masked image (areas to inpaint are zeroed out)
        masked_image = image_tensor * (1 - mask_tensor)
        
        return {
            'prompt': prompt,
            'image': image_tensor,
            'mask': mask_tensor,
            'masked_image': masked_image,
            'image_pil': source_image_resized,  # PIL Image for pipeline
            'mask_pil': mask_resized,  # PIL Image for pipeline
            'image_path': image_path,
            'mask_path': mask_path,
        }


def sd_inpainting_collate_function(batch):
    """Collate function for SD Inpainting dataset."""
    prompts = [item['prompt'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    masked_images = torch.stack([item['masked_image'] for item in batch])
    images_pil = [item['image_pil'] for item in batch]  # Keep PIL images
    masks_pil = [item['mask_pil'] for item in batch]  # Keep PIL masks
    image_paths = [item['image_path'] for item in batch]
    mask_paths = [item['mask_path'] for item in batch]
    
    return {
        'prompts': prompts,
        'images': images,
        'masks': masks,
        'masked_images': masked_images,
        'images_pil': images_pil,
        'masks_pil': masks_pil,
        'image_paths': image_paths,
        'mask_paths': mask_paths,
    }


class SDInpaintingLatentDataset(Dataset):
    """Dataset with pre-computed latents for SD1.5 Inpainting.
    
    This dataset expects pre-processed embeddings stored in subdirectories.
    """
    
    def __init__(
        self,
        json_path: str,
        num_latent_t: int = 1,
        cfg_rate: float = 0.0,
    ):
        self.json_path = json_path
        self.cfg_rate = cfg_rate
        self.data_dir = os.path.dirname(json_path)
        
        # Directories for pre-computed embeddings
        self.prompt_embed_dir = os.path.join(self.data_dir, "prompt_embed")
        self.source_latents_dir = os.path.join(self.data_dir, "source_latents")
        self.mask_dir = os.path.join(self.data_dir, "masks_processed")
        self.masked_latents_dir = os.path.join(self.data_dir, "masked_latents")
        
        # Load data annotations
        with open(json_path, 'r') as f:
            self.data_anno = json.load(f)
        
        self.num_latent_t = num_latent_t
        self.cfg_rate = cfg_rate
        
        self.lengths = [
            data_item.get("length", 1) for data_item in self.data_anno
        ]
    
    def __len__(self):
        return len(self.data_anno)
    
    def __getitem__(self, idx):
        item = self.data_anno[idx]
        
        # Load prompt embeddings
        prompt_embed_file = item.get("prompt_embed_path", None)
        source_latents_file = item.get("source_latents_path", None)
        mask_file = item.get("mask_path", None)
        masked_latents_file = item.get("masked_latents_path", None)
        
        prompt_embed = None
        if prompt_embed_file:
            prompt_embed = torch.load(
                os.path.join(self.prompt_embed_dir, prompt_embed_file),
                map_location="cpu",
                weights_only=True,
            )
        
        source_latents = None
        if source_latents_file:
            source_latents = torch.load(
                os.path.join(self.source_latents_dir, source_latents_file),
                map_location="cpu",
                weights_only=True,
            )
        
        mask = None
        if mask_file:
            mask = torch.load(
                os.path.join(self.mask_dir, mask_file),
                map_location="cpu",
                weights_only=True,
            )
        
        masked_latents = None
        if masked_latents_file:
            masked_latents = torch.load(
                os.path.join(self.masked_latents_dir, masked_latents_file),
                map_location="cpu",
                weights_only=True,
            )
        
        caption = item['caption']
        
        return (
            prompt_embed,
            source_latents,
            mask,
            masked_latents,
            caption,
        )


def sd_inpainting_latent_collate_function(batch):
    """Collate function for pre-computed latent dataset."""
    (
        prompt_embeds,
        source_latents,
        masks,
        masked_latents,
        captions,
    ) = zip(*batch)
    
    # Handle None values
    if prompt_embeds[0] is not None:
        prompt_embeds = torch.stack(prompt_embeds, dim=0)
    else:
        prompt_embeds = None
    
    if source_latents[0] is not None:
        source_latents = torch.stack(source_latents, dim=0)
    else:
        source_latents = None
    
    if masks[0] is not None:
        masks = torch.stack(masks, dim=0)
    else:
        masks = None
    
    if masked_latents[0] is not None:
        masked_latents = torch.stack(masked_latents, dim=0)
    else:
        masked_latents = None
    
    return (
        prompt_embeds,
        source_latents,
        masks,
        masked_latents,
        captions,
    )


if __name__ == "__main__":
    # Test the dataset
    dataset = SDInpaintingDataset("data/test_metadata.jsonl", image_size=512)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Prompt: {sample['prompt']}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
