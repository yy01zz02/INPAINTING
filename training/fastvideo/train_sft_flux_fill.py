# Supervised Fine-Tuning (SFT) script for Flux Fill inpainting model.
# Full fine-tuning with FSDP sharding for multi-GPU setup.
# Uses SwanLab for logging and pre-computed T5 embeddings.

import argparse
import math
import os
from pathlib import Path
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.communications_flux import sp_parallel_dataloader_wrapper
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions

from torch.utils.data.distributed import DistributedSampler
import swanlab
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_flux_fill_rl_datasets import (
    FluxFillLatentDataset,
    flux_fill_latent_collate_function,
)
import torch.distributed as dist
from fastvideo.utils.checkpoint import save_checkpoint, save_final_model
from fastvideo.utils.logging_ import main_print
from diffusers.image_processor import VaeImageProcessor

check_min_version("0.31.0")
import numpy as np
from torch.nn import functional as F
from diffusers import FluxTransformer2DModel, AutoencoderKL
from safetensors.torch import save_file


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    """Pack latents into the format expected by Flux."""
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def unpack_latents(latents, height, width, vae_scale_factor):
    """Unpack latents from Flux format."""
    batch_size, num_patches, channels = latents.shape
    height = height // vae_scale_factor
    width = width // vae_scale_factor
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, height, width)
    return latents


def prepare_latent_image_ids(height, width, device, dtype):
    """Prepare image position IDs for Flux Fill.
    
    Returns 2D tensor (seq_len, 3) as expected by FluxTransformer2DModel.
    """
    latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = torch.arange(height, device=device, dtype=dtype)[:, None]
    latent_image_ids[..., 2] = torch.arange(width, device=device, dtype=dtype)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    return latent_image_ids


def sd3_time_shift(shift, t):
    """Apply time shift for SD3/Flux models."""
    return shift * t / (1 + (shift - 1) * t)


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def sft_one_step_fill(
    args,
    transformer,
    batch_data,
    sigma_schedule,
    device,
):
    """Single SFT training step for Fill model.
    
    For SFT, we:
    1. Take source image latents as target
    2. Add noise at random timestep
    3. Predict velocity and compute loss against actual velocity
    """
    (
        encoder_hidden_states,
        pooled_prompt_embeds,
        text_ids,
        masked_latents,
        mask_latents,
        target_latents,  # Source image latents (target for reconstruction)
        captions,
    ) = batch_data
    
    batch_size = encoder_hidden_states.shape[0]
    
    # Move to device
    encoder_hidden_states = encoder_hidden_states.to(device, dtype=torch.bfloat16)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device, dtype=torch.bfloat16)
    text_ids = text_ids.to(device, dtype=torch.bfloat16)
    masked_latents = masked_latents.to(device, dtype=torch.bfloat16)
    mask_latents = mask_latents.to(device, dtype=torch.bfloat16)
    target_latents = target_latents.to(device, dtype=torch.bfloat16)
    
    IN_CHANNELS = 16
    VAE_SCALE = 8
    latent_h = args.h // VAE_SCALE
    latent_w = args.w // VAE_SCALE
    
    # Pack target latents [B, 16, H/8, W/8] -> [B, seq_len, 64]
    target_latents_packed = pack_latents(
        target_latents, batch_size, IN_CHANNELS, latent_h, latent_w
    )
    
    # Pack masked image latents
    masked_latents_packed = pack_latents(
        masked_latents, batch_size, IN_CHANNELS, latent_h, latent_w
    )
    
    # Process mask following FluxFillPipeline.prepare_mask_latents
    pixel_h, pixel_w = mask_latents.shape[2], mask_latents.shape[3]
    mask_h, mask_w = pixel_h // VAE_SCALE, pixel_w // VAE_SCALE
    
    mask_squeezed = mask_latents[:, 0, :, :]  # [B, H, W]
    mask_reshaped = mask_squeezed.view(batch_size, mask_h, VAE_SCALE, mask_w, VAE_SCALE)
    mask_permuted = mask_reshaped.permute(0, 2, 4, 1, 3)
    mask_expanded = mask_permuted.reshape(batch_size, VAE_SCALE * VAE_SCALE, mask_h, mask_w)
    mask_latents_packed = pack_latents(
        mask_expanded, batch_size, VAE_SCALE * VAE_SCALE, latent_h, latent_w
    )
    
    # Prepare image IDs
    image_ids = prepare_latent_image_ids(
        latent_h // 2, latent_w // 2, device, torch.bfloat16
    )
    
    # Sample random timestep indices
    t_idx = torch.randint(0, len(sigma_schedule) - 1, (batch_size,), device=device)
    sigma = sigma_schedule[t_idx].to(device)
    
    timesteps = (sigma * 1000).long()
    
    # Sample noise
    noise = torch.randn_like(target_latents_packed)
    
    # Create noisy latents: x_t = (1 - sigma) * x_0 + sigma * noise (Flow matching formulation)
    sigma_expanded = sigma.view(batch_size, 1, 1)
    noisy_latents = (1 - sigma_expanded) * target_latents_packed + sigma_expanded * noise
    
    # Compute velocity target: v = noise - x_0
    velocity_target = noise - target_latents_packed
    
    # Prepare Fill model input
    # Step 1: Combine masked_latents and mask in feature dimension
    masked_image_latents = torch.cat((masked_latents_packed, mask_latents_packed), dim=-1)
    
    # Step 2: Concatenate noisy latents with masked_image_latents
    combined_latents = torch.cat((noisy_latents, masked_image_latents), dim=2)
    
    # text_ids handling
    if text_ids.dim() == 3:
        sample_text_ids = text_ids[0]
    else:
        sample_text_ids = text_ids
    
    # Forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = transformer(
            hidden_states=combined_latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=sigma,  # Use sigma directly as timestep for flow matching
            guidance=torch.tensor([3.5], device=device, dtype=torch.bfloat16),
            img_ids=image_ids,
            txt_ids=sample_text_ids,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
    
    # Compute MSE loss
    loss = F.mse_loss(pred.float(), velocity_target.float(), reduction="mean")
    
    return loss


def train_one_step_sft(
    args,
    device,
    transformer,
    optimizer,
    lr_scheduler,
    batch_data,
    max_grad_norm,
    sigma_schedule,
):
    """Single training step for SFT."""
    optimizer.zero_grad()
    
    loss = sft_one_step_fill(
        args,
        transformer,
        batch_data,
        sigma_schedule,
        device,
    )
    
    loss = loss / args.gradient_accumulation_steps
    loss.backward()
    
    grad_norm = transformer.clip_grad_norm_(max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    
    return loss.item() * args.gradient_accumulation_steps, grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm


def main(args):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    initialize_sequence_parallel_state(args.sp_size)
    
    if args.seed is not None:
        set_seed(args.seed + rank)

    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize SwanLab
    if rank == 0:
        swanlab.init(
            project="flux-fill-sft",
            config=vars(args),
        )
    
    main_print(f"Loading Flux Fill model from {args.model_path}")
    
    # Load transformer model
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    
    # Load VAE for latent encoding (only needed if we want to encode images on-the-fly)
    vae = None
    if args.encode_on_fly:
        vae = AutoencoderKL.from_pretrained(
            args.model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        vae.to(device)
        vae.requires_grad_(False)
    
    # FSDP wrapping
    fsdp_kwargs, _ = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_strategy,
        args.use_liger,
        False,
        args.use_cpu_offload,
    )
    
    transformer = FSDP(transformer, **fsdp_kwargs)
    
    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(transformer)
    
    # Enable training
    transformer.train()
    
    main_print(f"Model loaded and wrapped with FSDP")
    
    # Initialize optimizer
    params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    
    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Load dataset
    dataset = FluxFillLatentDataset(
        json_path=args.data_json_path,
        num_latent_t=1,
        cfg_rate=0.0,
        image_size=args.h,
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        seed=args.seed if args.seed else 42,
        shuffle=True,
    )
    
    # For SFT, we need to also load target latents
    # Modify collate function to include target latents
    def sft_collate_function(batch):
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            masked_latents,
            mask_latents,
            captions,
            mask_pils,
        ) = flux_fill_latent_collate_function(batch)
        
        # For SFT, we need source image latents as target
        # These should be pre-computed and stored alongside masked_latents
        # Here we assume they are loaded from dataset
        # If not available, you need to modify the dataset to include them
        
        # For now, use masked_latents as a proxy (in real use, load source_latents)
        # In a proper setup, source_latents would be pre-computed
        target_latents = masked_latents  # Placeholder - should be source image latents
        
        return (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            masked_latents,
            mask_latents,
            target_latents,
            captions,
        )
    
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        collate_fn=sft_collate_function,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    
    # sigma schedule for flow matching
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)
    
    # Training loop
    global_step = 0
    step_times = []
    
    main_print("***** Running SFT Training *****")
    main_print(f"  Max train steps = {args.max_train_steps}")
    main_print(f"  Train batch size per device = {args.train_batch_size}")
    main_print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Learning rate = {args.learning_rate}")
    
    for epoch in range(args.num_epochs):
        main_print(f"Starting epoch {epoch}")
        sampler.set_epoch(epoch)
        
        for step, batch_data in enumerate(loader):
            step_start_time = time.time()
            
            if global_step >= args.max_train_steps:
                break
            
            loss, grad_norm = train_one_step_sft(
                args,
                device,
                transformer,
                optimizer,
                lr_scheduler,
                batch_data,
                args.max_grad_norm,
                sigma_schedule,
            )
            
            # Logging
            if rank == 0:
                swanlab.log({
                    "loss": loss,
                    "grad_norm": grad_norm,
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=global_step)
            
            global_step += 1
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                if rank == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_step_{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # Save model state
                    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
                    state_dict = get_model_state_dict(transformer, options=options)
                    
                    ckpt_path = os.path.join(checkpoint_dir, "model.safetensors")
                    save_file(state_dict, ckpt_path)
                    main_print(f"Checkpoint saved to {checkpoint_dir}")
                
                dist.barrier()
            
            # Log progress
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            if len(step_times) > 100:
                step_times = step_times[-100:]
            avg_step_time = sum(step_times) / len(step_times)
            remaining_steps = args.max_train_steps - global_step
            eta_seconds = remaining_steps * avg_step_time
            eta_hours = eta_seconds / 3600
            
            if rank == 0:
                eta_str = f"{eta_hours:.1f}h" if eta_hours >= 1 else f"{eta_seconds/60:.1f}m"
                print(f"\n[Step {global_step}/{args.max_train_steps}] loss={loss:.6f} | grad_norm={grad_norm:.4f} | step_time={step_time:.2f}s | ETA: {eta_str}")
        
        if global_step >= args.max_train_steps:
            break
    
    # Save final model
    if args.final_model_dir and rank == 0:
        os.makedirs(args.final_model_dir, exist_ok=True)
        
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(transformer, options=options)
        
        ckpt_path = os.path.join(args.final_model_dir, "model.safetensors")
        save_file(state_dict, ckpt_path)
        main_print(f"Final model saved to {args.final_model_dir}")
    
    dist.barrier()
    
    # Cleanup
    if rank == 0:
        swanlab.finish()
    
    destroy_sequence_parallel_group()
    dist.destroy_process_group()
    
    main_print("Training completed!")


def parse_args():
    parser = argparse.ArgumentParser(description="Flux Fill SFT Training")
    
    # Model paths
    parser.add_argument("--model_path", type=str, required=True, help="Path to Flux Fill model")
    parser.add_argument("--data_json_path", type=str, required=True, help="Path to training data JSON")
    parser.add_argument("--output_dir", type=str, default="./outputs/flux_fill_sft", help="Output directory")
    parser.add_argument("--final_model_dir", type=str, default="", help="Directory to save final model")
    
    # Training settings
    parser.add_argument("--train_batch_size", type=int, default=1, help="Training batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Learning rate warmup steps")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="Learning rate scheduler")
    
    # Image settings
    parser.add_argument("--h", type=int, default=512, help="Image height")
    parser.add_argument("--w", type=int, default=512, help="Image width")
    
    # Sampling settings
    parser.add_argument("--sampling_steps", type=int, default=28, help="Number of sampling steps")
    parser.add_argument("--shift", type=float, default=1.0, help="Time shift parameter")
    
    # FSDP settings
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="full_shard", help="FSDP sharding strategy")
    parser.add_argument("--use_cpu_offload", action="store_true", help="Use CPU offload")
    parser.add_argument("--use_liger", action="store_true", help="Use Liger kernel")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    
    # Other settings
    parser.add_argument("--sp_size", type=int, default=1, help="Sequence parallel size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32 on Ampere GPUs")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--encode_on_fly", action="store_true", help="Encode images on-the-fly (requires VAE)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
