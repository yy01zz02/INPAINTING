"""
PowerPaint v2 inference script for RLBench evaluation.
Generates inpainted images using PowerPaint v2 baseline model.
"""
import torch
import os
from PIL import Image
import argparse
from transformers import CLIPTextModel
from safetensors.torch import load_model
from diffusers import UniPCMultistepScheduler

from PowerPaint.powerpaint.models.BrushNet_CA import BrushNetModel
from PowerPaint.powerpaint.models.unet_2d_condition import UNet2DConditionModel
from PowerPaint.powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
from PowerPaint.powerpaint.utils.utils import TokenizerWrapper, add_tokens

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser(description="PowerPaint v2 inference")
    parser.add_argument('--checkpoint_dir', type=str, default="",
                        help="Path to PowerPaint v2 checkpoint directory")
    parser.add_argument('--sd15_path', type=str, default="",
                        help="Path to SD1.5 model (for UNet and text encoder)")
    parser.add_argument('--image_save_path', type=str, default="./results/powerpaint",
                        help="Directory to save generated images")
    parser.add_argument('--base_dir', type=str, default="./data/rlbench",
                        help="Base directory containing masks, masked_images, prompts")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--fitting_degree', type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16

    print("Loading PowerPaint v2 model...")
    
    # Load base UNet and text encoder from SD1.5
    unet = UNet2DConditionModel.from_pretrained(
        args.sd15_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    
    text_encoder_brushnet = CLIPTextModel.from_pretrained(
        args.sd15_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    
    brushnet = BrushNetModel.from_unet(unet)
    base_model_path = os.path.join(args.checkpoint_dir, "realisticVisionV60B1_v51VAE")
    
    pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
        base_model_path,
        brushnet=brushnet,
        text_encoder_brushnet=text_encoder_brushnet,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=False,
        safety_checker=None,
    )
    
    pipe.unet = UNet2DConditionModel.from_pretrained(
        base_model_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    
    # Load PowerPaint checkpoint weights
    load_model(pipe.brushnet, os.path.join(args.checkpoint_dir, "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"))
    
    # Setup tokenizer with special tokens
    pipe.tokenizer = TokenizerWrapper(
        from_pretrained=base_model_path,
        subfolder="tokenizer",
        revision=None,
    )
    add_tokens(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder_brushnet,
        placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
        initialize_tokens=["a"]
    )
    
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
                num_inference_steps=args.ddim_steps,
                generator=generator,
                guidance_scale=args.guidance_scale,
                fitting_degree=args.fitting_degree,
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
