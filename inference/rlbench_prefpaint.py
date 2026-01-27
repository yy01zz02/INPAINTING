"""
PrefPaint Baseline Inference Script for RLBench Evaluation.

This script runs inference using PrefPaint model for inpainting benchmark evaluation.
PrefPaint is a preference-based inpainting model that serves as a baseline comparison.

Usage:
    export PREFPAINT_MODEL_PATH="/path/to/prefpaint/model"
    export RLBENCH_DIR="/path/to/rlbench/dataset"
    export OUTPUT_DIR="/path/to/output"
    
    python rlbench_prefpaint.py
"""

import os
import argparse

import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PrefPaint Inpainting Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("PREFPAINT_MODEL_PATH", ""),
        help="Path to PrefPaint model checkpoint",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=os.environ.get("RLBENCH_DIR", ""),
        help="Path to RLBench dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "results/prefpaint"),
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def load_model(model_path: str, device: str):
    """Load PrefPaint model.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded pipeline
    """
    print(f"Loading PrefPaint model from {model_path}")
    
    pipe = AutoPipelineForInpainting.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)
    
    # Disable safety checker for benchmark evaluation
    pipe.safety_checker = None
    if hasattr(pipe, "feature_extractor"):
        pipe.feature_extractor = None
    
    return pipe


def run_inference(
    pipe,
    masked_image: Image.Image,
    mask_image: Image.Image,
    prompt: str,
    seed: int,
    device: str,
) -> Image.Image:
    """Run inpainting inference.
    
    Args:
        pipe: Diffusion pipeline
        masked_image: Input image with masked region
        mask_image: Binary mask (white = inpaint region)
        prompt: Text prompt for generation
        seed: Random seed
        device: Device string
        
    Returns:
        Generated image
    """
    width, height = masked_image.size
    generator = torch.Generator(device).manual_seed(seed)
    
    result = pipe(
        prompt=prompt,
        image=masked_image,
        mask_image=mask_image,
        generator=generator,
        height=height,
        width=width,
    )
    
    return result.images[0]


def main():
    """Main inference function."""
    args = parse_args()
    
    # Validate arguments
    if not args.model_path:
        raise ValueError("Please set PREFPAINT_MODEL_PATH environment variable")
    if not args.base_dir:
        raise ValueError("Please set RLBENCH_DIR environment variable")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    pipe = load_model(args.model_path, device)
    
    # Setup directories
    masks_dir = os.path.join(args.base_dir, "masks")
    masked_images_dir = os.path.join(args.base_dir, "masked_images")
    prompts_dir = os.path.join(args.base_dir, "prompts")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all mask files
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(".jpg")])
    print(f"Found {len(mask_files)} images to process")
    print(f"Saving results to {args.output_dir}")
    
    # Process each image
    for idx, mask_file in enumerate(mask_files):
        file_name = os.path.splitext(mask_file)[0]
        
        # Build file paths
        mask_path = os.path.join(masks_dir, mask_file)
        masked_image_path = os.path.join(masked_images_dir, f"{file_name}.jpg")
        prompt_path = os.path.join(prompts_dir, f"{file_name}.txt")
        save_path = os.path.join(args.output_dir, f"{file_name}.jpg")
        
        # Check required files exist
        if not os.path.exists(masked_image_path):
            print(f"Warning: Masked image not found for {file_name}, skipping...")
            continue
        
        if not os.path.exists(prompt_path):
            print(f"Warning: Prompt file not found for {file_name}, skipping...")
            continue
        
        # Skip if already generated
        if os.path.exists(save_path):
            print(f"[{idx+1}/{len(mask_files)}] {file_name} exists, skipping...")
            continue
        
        print(f"[{idx+1}/{len(mask_files)}] Processing {file_name}...")
        
        try:
            # Load inputs
            masked_image = Image.open(masked_image_path).convert("RGB")
            mask_image = Image.open(mask_path).convert("L")
            
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            
            width, height = masked_image.size
            print(f"  Size: {width}x{height}, Prompt: {prompt[:80]}...")
            
            # Generate
            result = run_inference(
                pipe=pipe,
                masked_image=masked_image,
                mask_image=mask_image,
                prompt=prompt,
                seed=args.seed,
                device=device,
            )
            
            # Save result
            result.save(save_path)
            print(f"  Saved: {save_path}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print("Inference complete!")


if __name__ == "__main__":
    main()
