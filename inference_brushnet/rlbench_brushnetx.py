"""
BrushNetX inference script for RLBench evaluation.
Generates inpainted images using BrushNetX (BrushEdit) baseline model.
"""
import torch
import os
from PIL import Image
import argparse
from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="BrushNetX inference")
    parser.add_argument('--brushnet_path', type=str, default="",
                        help="Path to BrushNetX model")
    parser.add_argument('--base_model_path', type=str, default="",
                        help="Path to base Stable Diffusion model")
    parser.add_argument('--image_save_path', type=str, default="./results/brushnetx",
                        help="Directory to save generated images")
    parser.add_argument('--base_dir', type=str, default="./data/rlbench",
                        help="Base directory containing masks, masked_images, prompts")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")
    parser.add_argument('--num_inference_steps', type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load BrushNet model
    print(f"Loading BrushNetX from {args.brushnet_path}")
    brushnet = BrushNetModel.from_pretrained(args.brushnet_path, torch_dtype=torch.float16)
    
    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        args.base_model_path,
        brushnet=brushnet,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False
    )
    pipe.safety_checker = None
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    print("Model loaded successfully!")

    # Setup directories
    masks_dir = os.path.join(args.base_dir, "masks")
    masked_images_dir = os.path.join(args.base_dir, "masked_images")
    prompts_dir = os.path.join(args.base_dir, "prompts")
    os.makedirs(args.image_save_path, exist_ok=True)

    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.jpg')])
    print(f"Found {len(mask_files)} images to process")

    for idx, mask_file in enumerate(mask_files):
        file_name = os.path.splitext(mask_file)[0]
        mask_path = os.path.join(masks_dir, mask_file)
        masked_image_path = os.path.join(masked_images_dir, f"{file_name}.jpg")
        prompt_path = os.path.join(prompts_dir, f"{file_name}.txt")
        save_path = os.path.join(args.image_save_path, f"{file_name}.jpg")

        if not os.path.exists(masked_image_path) or not os.path.exists(prompt_path):
            continue
        if os.path.exists(save_path):
            print(f"[{idx+1}/{len(mask_files)}] {file_name} exists, skipping...")
            continue

        print(f"[{idx+1}/{len(mask_files)}] Generating {file_name}...")

        try:
            masked_image = Image.open(masked_image_path).convert("RGB")
            mask_image = Image.open(mask_path).convert("L")
            width, height = masked_image.size

            with open(prompt_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()

            generator = torch.Generator(device).manual_seed(args.seed)
            image = pipe(
                prompt=caption,
                image=masked_image,
                mask=mask_image,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                height=height,
                width=width,
            ).images[0]

            image.save(save_path)
            print(f"  Saved: {save_path}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print("Generation complete.")


if __name__ == "__main__":
    main()
