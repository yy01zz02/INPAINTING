"""
Flux.1 Fill + GRPO inference script for RLBench evaluation.
Generates inpainted images using the GRPO fine-tuned model (loads EMA weights).
"""
import torch
import os
from PIL import Image
import argparse
from diffusers import FluxFillPipeline
from safetensors.torch import load_file


def parse_args():
    parser = argparse.ArgumentParser(description="Flux.1 Fill + GRPO inference")
    parser.add_argument('--model_path', type=str, default="",
                        help="Path to base Flux.1 Fill model")
    parser.add_argument('--finetuned_model_path', type=str, default="",
                        help="Path to GRPO fine-tuned checkpoint (EMA weights)")
    parser.add_argument('--image_save_path', type=str, default="./results/fill_grpo",
                        help="Directory to save generated images")
    parser.add_argument('--base_dir', type=str, default="./data/rlbench",
                        help="Base directory containing masks, masked_images, prompts")
    parser.add_argument('--seed', type=int, default=1936, help="Random seed")
    parser.add_argument('--num_inference_steps', type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load base model
    print(f"Loading Flux.1 Fill model from {args.model_path}")
    pipe = FluxFillPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)

    # Load GRPO fine-tuned weights (EMA)
    if args.finetuned_model_path:
        print(f"Loading GRPO fine-tuned weights from {args.finetuned_model_path}")
        weights_path = os.path.join(args.finetuned_model_path, "ema_state_dict.safetensors")
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)
            pipe.transformer.load_state_dict(state_dict, strict=True)
            print("Successfully loaded GRPO fine-tuned weights")
        else:
            print(f"Warning: Weights not found at {weights_path}, using base model")

    pipe = pipe.to(device)
    pipe.safety_checker = None

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
                mask_image=mask_image,
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
