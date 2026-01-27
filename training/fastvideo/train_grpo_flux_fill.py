# Training script for Flux Fill inpainting model with GRPO.
# Supports full fine-tuning with FSDP sharding for 4x96GB GPU setup.
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
from fastvideo.rewards import get_reward_fn
import torch.distributed as dist
from fastvideo.utils.checkpoint import save_checkpoint, save_final_model
from fastvideo.utils.logging_ import main_print
from diffusers.image_processor import VaeImageProcessor

check_min_version("0.31.0")
from collections import deque
import numpy as np
from torch.nn import functional as F
from typing import List
from PIL import Image
from diffusers import FluxTransformer2DModel, AutoencoderKL
from contextlib import contextmanager
from safetensors.torch import save_file


class FSDP_EMA:
    """Exponential Moving Average handler for FSDP models."""
    
    def __init__(self, model, decay, rank):
        self.decay = decay
        self.rank = rank
        self.ema_state_dict_rank0 = {}
        
        # All ranks must participate in get_model_state_dict (collective operation)
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(model, options=options)

        # But only rank 0 stores the EMA weights
        if self.rank == 0:
            self.ema_state_dict_rank0 = {k: v.clone() for k, v in state_dict.items()}
            main_print("--> Modern EMA handler initialized on rank 0.")

    def update(self, model):
        # All ranks must participate in get_model_state_dict (collective operation)
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state_dict = get_model_state_dict(model, options=options)

        # But only rank 0 updates the EMA weights
        if self.rank == 0:
            for key in self.ema_state_dict_rank0:
                if key in model_state_dict:
                    self.ema_state_dict_rank0[key].copy_(
                        self.decay * self.ema_state_dict_rank0[key] + (1 - self.decay) * model_state_dict[key]
                    )

    @contextmanager
    def use_ema_weights(self, model):
        backup_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        backup_state_dict_rank0 = get_model_state_dict(model, options=backup_options)

        load_options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        set_model_state_dict(
            model,
            model_state_dict=self.ema_state_dict_rank0,
            options=load_options
        )
        
        try:
            yield
        finally:
            restore_options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
            set_model_state_dict(
                model,
                model_state_dict=backup_state_dict_rank0,
                options=restore_options
            )


def save_ema_checkpoint(ema_handler, rank, output_dir, step, epoch, config_dict):
    """Save EMA checkpoint."""
    if rank == 0:
        save_dir = os.path.join(output_dir, f"ema_checkpoint_{step}")
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, "ema_state_dict.safetensors")
        save_file(ema_handler.ema_state_dict_rank0, ckpt_path)
        main_print(f"--> EMA checkpoint saved to {save_dir}")


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


def flux_step(pred, latents, eta, sigma_schedule, step, prev_sample=None, grpo=False, sde_solver=False):
    """Single Flux sampling step with log probability calculation."""
    sigma_curr = sigma_schedule[step]
    sigma_next = sigma_schedule[step + 1]
    
    if sde_solver:
        dt = sigma_next - sigma_curr
        dw = torch.randn_like(latents) * ((-dt).sqrt())
        
        pred_original = latents - sigma_curr * pred
        next_sample = latents + pred * dt + eta * dw
        
        noise = (next_sample - latents - pred * dt) / (eta + 1e-8)
        log_prob = -0.5 * (noise ** 2).sum(dim=tuple(range(1, noise.ndim)))
    else:
        dt = sigma_next - sigma_curr
        pred_original = latents - sigma_curr * pred
        next_sample = latents + pred * dt
        log_prob = torch.zeros(latents.shape[0], device=latents.device)
    
    if grpo:
        return next_sample, pred_original, log_prob
    return next_sample


def run_sample_step_fill(
    args,
    input_latents,
    progress_bar,
    sigma_schedule,
    transformer,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    image_ids,
    masked_latents,
    mask_latents,
    grpo_sample=False,
):
    """Run sampling steps for Fill (inpainting) model.
    
    For Flux Fill, the hidden_states are constructed as follows (from diffusers):
    1. masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)  
       - This concatenates the mask to the masked image latents in feature dimension
    2. hidden_states = torch.cat((latents, masked_image_latents), dim=2)
       - This concatenates noisy latents with the conditioning in feature dimension
    
    The packed latent shape is [B, seq_len, features]:
    - latents: [B, seq_len, 64] (16 channels * 4 from packing)
    - masked_image_latents after mask concat: [B, seq_len, 64 + 256] = [B, seq_len, 320]
      (mask is packed from 64 channels to 256 features)
    - combined: [B, seq_len, 64 + 320] = [B, seq_len, 384]
    """
    latents = input_latents
    batch_latents = [latents.clone()]
    batch_log_probs = []
    
    for i in progress_bar:
        timestep_value = int(sigma_schedule[i] * 1000)
        timestep = torch.tensor([timestep_value], device=latents.device, dtype=torch.long)
        timestep = timestep.expand(latents.shape[0])
        
        # For Fill model: concatenate following diffusers FluxFillPipeline format
        # Step 1: Combine masked_latents and mask in feature dimension (dim=-1/dim=2)
        # masked_image_latents shape: [B, seq_len, 64], mask_latents shape: [B, seq_len, 256]
        # Result: [B, seq_len, 320]
        masked_image_latents = torch.cat((masked_latents, mask_latents), dim=-1)
        
        # Step 2: Concatenate noisy latents with masked_image_latents in feature dimension
        # latents shape: [B, seq_len, 64], masked_image_latents: [B, seq_len, 320]
        # Result: [B, seq_len, 384]
        combined_latents = torch.cat((latents, masked_image_latents), dim=2)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = transformer(
                hidden_states=combined_latents,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep / 1000,
                guidance=torch.tensor([3.5], device=latents.device, dtype=torch.bfloat16),
                img_ids=image_ids,
                txt_ids=text_ids,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
        
        # Prediction is for the noisy latents part only (first 64 features)
        # No need to slice since transformer output matches noisy latent shape
        
        z, pred_original, log_prob = flux_step(
            pred, latents.to(torch.float32), args.eta, sigma_schedule, i,
            prev_sample=None, grpo=True, sde_solver=True
        )
        
        latents = z.to(torch.bfloat16)
        batch_latents.append(latents.clone())
        batch_log_probs.append(log_prob)
    
    batch_latents = torch.stack(batch_latents, dim=1)
    batch_log_probs = torch.stack(batch_log_probs, dim=1)
    
    return z, latents, batch_latents, batch_log_probs


def grpo_one_step_fill(
    args,
    transformer,
    samples,
    sigma_schedule,
    train_timestep_idx,
):
    """Single GRPO training step for Fill model."""
    latents = samples["latents"][:, train_timestep_idx]
    next_latents = samples["next_latents"][:, train_timestep_idx]
    timesteps = samples["timesteps"][:, train_timestep_idx]
    encoder_hidden_states = samples["encoder_hidden_states"]
    pooled_prompt_embeds = samples["pooled_prompt_embeds"]
    text_ids = samples["text_ids"]
    image_ids = samples["image_ids"]
    masked_latents = samples["masked_latents"]
    mask_latents = samples["mask_latents"]
    
    batch_size = latents.shape[0]
    
    # For Fill model: concatenate following diffusers FluxFillPipeline format
    # Step 1: Combine masked_latents and mask in feature dimension (dim=-1/dim=2)
    masked_image_latents = torch.cat((masked_latents, mask_latents), dim=-1)
    
    # Step 2: Concatenate noisy latents with masked_image_latents in feature dimension
    combined_latents = torch.cat((latents, masked_image_latents), dim=2)
    
    # text_ids should be 2D (seq_len, dim) for transformer
    if text_ids.dim() == 3:
        sample_text_ids = text_ids[0]
    else:
        sample_text_ids = text_ids
    
    # image_ids should be 2D (seq_len, 3) for transformer
    if image_ids.dim() == 3:
        sample_image_ids = image_ids[0]
    else:
        sample_image_ids = image_ids
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = transformer(
            hidden_states=combined_latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps / 1000,
            guidance=torch.tensor([3.5], device=latents.device, dtype=torch.bfloat16),
            img_ids=sample_image_ids,
            txt_ids=sample_text_ids,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
    
    # Note: transformer output matches latents shape, no slicing needed
    # The transformer produces velocity prediction for the noisy latents
    
    step_idx = min(train_timestep_idx, len(sigma_schedule) - 2)
    sigma_curr = sigma_schedule[step_idx]
    sigma_next = sigma_schedule[step_idx + 1]
    dt = sigma_next - sigma_curr
    
    pred_next = latents.to(torch.float32) + pred.to(torch.float32) * dt
    
    noise = (next_latents.to(torch.float32) - pred_next) / (args.eta + 1e-8)
    log_prob = -0.5 * (noise ** 2).sum(dim=tuple(range(1, noise.ndim)))
    
    return log_prob


def sample_reference_model(
    args,
    device,
    transformer,
    vae,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    masked_latents,
    mask_latents,
    captions,
    reward_model,
    mask_pils=None,
):
    """Sample from reference model and compute rewards for Fill.
    
    Uses pre-computed T5 embeddings and latents.
    For inpainting reward, uses mask_pils for API call.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    w, h = args.w, args.h
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    IN_CHANNELS = 16
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1
    batch_indices = torch.chunk(torch.arange(B), max(1, B // batch_size))

    all_latents = []
    all_log_probs = []
    all_raw_scores = []  # Store raw scores for inpainting reward
    all_decoded_images = []  # Store decoded images for reward calculation
    all_image_ids = []
    all_encoder_hidden_states = []
    all_pooled_prompt_embeds = []
    all_text_ids = []
    all_masked_latents = []
    all_mask_latents = []

    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx].to(device)
        batch_pooled_prompt_embeds = pooled_prompt_embeds[batch_idx].to(device)
        batch_text_ids = text_ids[batch_idx].to(device)
        batch_masked_latents = masked_latents[batch_idx].to(device)
        batch_mask_latents = mask_latents[batch_idx].to(device)
        batch_captions = [captions[i] for i in batch_idx]
        
        # Pack masked image latents [B, 16, H/8, W/8] -> [B, seq_len, 64]
        masked_latents_packed = pack_latents(
            batch_masked_latents, len(batch_idx), IN_CHANNELS, latent_h, latent_w
        )
        
        # Process mask using official FluxFillPipeline.prepare_mask_latents method:
        # Input mask from preprocessing: [B, 1, H, W] at pixel resolution
        # Official reshape: [B, 1, H, W] -> [B, H/8, 8, W/8, 8] -> permute -> [B, 64, H/8, W/8]
        # This preserves edge details by keeping the 8x8 pixel block structure
        VAE_SCALE = 8
        batch_size_local = len(batch_idx)
        
        # batch_mask_latents shape: [B, 1, H, W] (pixel resolution from preprocessing)
        pixel_h, pixel_w = batch_mask_latents.shape[2], batch_mask_latents.shape[3]
        mask_h, mask_w = pixel_h // VAE_SCALE, pixel_w // VAE_SCALE
        
        # Official reshape following FluxFillPipeline.prepare_mask_latents:
        # mask: [B, 1, H, W] -> remove channel -> [B, H, W]
        mask_squeezed = batch_mask_latents[:, 0, :, :]  # [B, H, W]
        
        # Reshape: [B, H, W] -> [B, H/8, 8, W/8, 8]
        mask_reshaped = mask_squeezed.view(
            batch_size_local, mask_h, VAE_SCALE, mask_w, VAE_SCALE
        )
        
        # Permute: [B, H/8, 8, W/8, 8] -> [B, 8, 8, H/8, W/8]
        mask_permuted = mask_reshaped.permute(0, 2, 4, 1, 3)
        
        # Reshape to channel dimension: [B, 64, H/8, W/8]
        mask_expanded = mask_permuted.reshape(
            batch_size_local, VAE_SCALE * VAE_SCALE, mask_h, mask_w
        )
        
        # Pack mask: [B, 64, H/8, W/8] -> [B, seq_len, 256]
        mask_latents_packed = pack_latents(
            mask_expanded, batch_size_local, VAE_SCALE * VAE_SCALE, latent_h, latent_w
        )
        
        # Initialize noise
        if args.init_same_noise:
            input_latents = torch.randn(
                (1, IN_CHANNELS, latent_h, latent_w),
                device=device,
                dtype=torch.bfloat16,
            )
        else:
            input_latents = torch.randn(
                (len(batch_idx), IN_CHANNELS, latent_h, latent_w),
                device=device,
                dtype=torch.bfloat16,
            )
        
        input_latents_packed = pack_latents(
            input_latents, len(batch_idx), IN_CHANNELS, latent_h, latent_w
        )
        image_ids = prepare_latent_image_ids(
            latent_h // 2, latent_w // 2, device, torch.bfloat16
        )
        
        progress_bar = tqdm(range(0, sample_steps), desc=f"Sampling [{index+1}/{len(batch_indices)}]", leave=False, disable=local_rank > 0)
        
        if batch_text_ids.dim() == 3:
            sample_text_ids = batch_text_ids[0]
        else:
            sample_text_ids = batch_text_ids
        
        with torch.no_grad():
            z, latents, batch_latents, batch_log_probs = run_sample_step_fill(
                args,
                input_latents_packed,
                progress_bar,
                sigma_schedule,
                transformer,
                batch_encoder_hidden_states,
                batch_pooled_prompt_embeds,
                sample_text_ids,
                image_ids,
                masked_latents_packed,
                mask_latents_packed,  # Use packed mask latents
                grpo_sample=True,
            )
        
        all_image_ids.append(image_ids.unsqueeze(0))
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        all_masked_latents.append(masked_latents_packed)
        all_mask_latents.append(mask_latents_packed)  # Use packed mask latents
        all_encoder_hidden_states.append(batch_encoder_hidden_states)
        all_pooled_prompt_embeds.append(batch_pooled_prompt_embeds)
        all_text_ids.append(sample_text_ids.unsqueeze(0) if sample_text_ids.dim() == 2 else batch_text_ids)
        
        # Decode generated image
        vae.enable_tiling()
        image_processor = VaeImageProcessor(16)
        rank = int(os.environ.get("RANK", 0))
        
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded_latents = unpack_latents(latents, h, w, 8)
                decoded_latents = (decoded_latents / 0.3611) + 0.1159
                image = vae.decode(decoded_latents, return_dict=False)[0]
                decoded_image = image_processor.postprocess(image)
        
        image_path = f"./images/fill_{rank}_{index}.png"
        decoded_image[0].save(image_path)
        
        # Get mask for this sample
        batch_mask_pil = mask_pils[batch_idx[0]] if mask_pils is not None else None
        
        # Compute raw scores for inpainting reward
        if args.reward_type == 'inpainting' and batch_mask_pil is not None:
            raw_score = reward_model.compute_reward_raw(
                decoded_image[0], batch_mask_pil, batch_captions[0]
            )
            all_raw_scores.append(raw_score)
        else:
            # Fallback to simple reward for non-inpainting reward types
            reward = reward_model.compute_reward(
                decoded_image[0], batch_captions[0]
            )
            all_raw_scores.append({"reward": reward})
        
        all_decoded_images.append(decoded_image[0])
    
    # Compute final rewards
    if args.reward_type == 'inpainting' and len(all_raw_scores) > 0 and 'boundary_score' in all_raw_scores[0]:
        # Use group-normalized inpainting rewards
        all_rewards = reward_model.compute_group_normalized_rewards(
            all_raw_scores, args.num_generations
        )
    else:
        # Fallback: extract simple rewards
        all_rewards = torch.tensor(
            [s.get("reward", 0.0) for s in all_raw_scores], 
            device=device, 
            dtype=torch.float32
        )
    
    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_image_ids = torch.cat(all_image_ids, dim=0)
    all_masked_latents = torch.cat(all_masked_latents, dim=0)
    all_mask_latents = torch.cat(all_mask_latents, dim=0)
    all_encoder_hidden_states = torch.cat(all_encoder_hidden_states, dim=0)
    all_pooled_prompt_embeds = torch.cat(all_pooled_prompt_embeds, dim=0)
    all_text_ids = torch.cat(all_text_ids, dim=0)
    
    return (
        all_rewards, all_latents, all_log_probs, sigma_schedule,
        all_image_ids, all_masked_latents, all_mask_latents,
        all_encoder_hidden_states, all_pooled_prompt_embeds, all_text_ids
    )


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def train_one_step(
    args,
    device,
    transformer,
    vae,
    reward_model,
    optimizer,
    lr_scheduler,
    batch_data,
    max_grad_norm,
    ema_handler,
):
    """Single training step for Fill model using pre-computed embeddings."""
    total_loss = 0.0
    optimizer.zero_grad()
    
    # Unpack batch data
    (
        encoder_hidden_states,
        pooled_prompt_embeds,
        text_ids,
        masked_latents,
        mask_latents,
        captions,
        mask_pils,
    ) = batch_data
    
    if args.use_group:
        B = encoder_hidden_states.shape[0]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(args.num_generations, dim=0)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(args.num_generations, dim=0)
        text_ids = text_ids.repeat_interleave(args.num_generations, dim=0)
        masked_latents = masked_latents.repeat_interleave(args.num_generations, dim=0)
        mask_latents = mask_latents.repeat_interleave(args.num_generations, dim=0)
        captions = [c for c in captions for _ in range(args.num_generations)]
        # Expand mask_pils for multiple generations
        mask_pils = [m for m in mask_pils for _ in range(args.num_generations)]
    
    (
        reward, all_latents, all_log_probs, sigma_schedule,
        all_image_ids, all_masked_latents, all_mask_latents,
        all_encoder_hidden_states, all_pooled_prompt_embeds, all_text_ids
    ) = sample_reference_model(
        args,
        device,
        transformer,
        vae,
        encoder_hidden_states,
        pooled_prompt_embeds,
        text_ids,
        masked_latents,
        mask_latents,
        captions,
        reward_model,
        mask_pils,
    )
    
    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    timesteps = torch.tensor(timestep_values, device=device, dtype=torch.long)
    
    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[:, :-1],
        "next_latents": all_latents[:, 1:],
        "log_probs": all_log_probs,
        "rewards": reward.to(torch.float32),
        "image_ids": all_image_ids,
        "masked_latents": all_masked_latents,
        "mask_latents": all_mask_latents,
        "encoder_hidden_states": all_encoder_hidden_states,
        "pooled_prompt_embeds": all_pooled_prompt_embeds,
        "text_ids": all_text_ids,
    }
    
    gathered_reward = gather_tensor(samples["rewards"])
    if dist.get_rank() == 0:
        print("gathered_reward", gathered_reward)
        with open('./reward.txt', 'a') as f:
            f.write(f"{gathered_reward.mean().item()}\n")
    
    # Compute advantages
    if args.use_group:
        n = len(samples["rewards"]) // args.num_generations
        advantages = torch.zeros_like(samples["rewards"])
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = samples["rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        samples["advantages"] = advantages
    else:
        advantages = (samples["rewards"] - gathered_reward.mean()) / (gathered_reward.std() + 1e-8)
        samples["advantages"] = advantages
    
    samples["final_advantages"] = torch.clamp(samples["advantages"], -args.adv_clip_max, args.adv_clip_max)
    
    train_timesteps = int(args.timestep_fraction * args.sampling_steps)
    grad_norm = 0.0
    
    for t_idx in range(train_timesteps):
        perm = torch.randperm(batch_size, device=device)
        
        for mini_batch_start in range(0, batch_size, args.train_batch_size):
            mini_batch_end = min(mini_batch_start + args.train_batch_size, batch_size)
            mini_batch_idx = perm[mini_batch_start:mini_batch_end]
            
            mini_samples = {k: v[mini_batch_idx] if isinstance(v, torch.Tensor) else v for k, v in samples.items()}
            
            log_prob = grpo_one_step_fill(
                args,
                transformer,
                mini_samples,
                sigma_schedule,
                t_idx,
            )
            
            old_log_prob = mini_samples["log_probs"][:, t_idx]
            advantages = mini_samples["final_advantages"]
            
            ratio = torch.exp(log_prob - old_log_prob)
            
            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio, 1.0 - args.clip_range, 1.0 + args.clip_range
            )
            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
            loss = loss / (args.gradient_accumulation_steps * train_timesteps)
            
            loss.backward()
            total_loss += loss.item()
    
    grad_norm = transformer.clip_grad_norm_(max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    
    reward_mean = gathered_reward.mean().item()
    return total_loss, grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm, reward_mean


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
        os.makedirs("./images", exist_ok=True)

    reward_model = get_reward_fn(args.reward_type, device)
    main_print(f"--> Using reward model: {args.reward_type}")

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    
    transformer = FSDP(transformer, **fsdp_kwargs)

    ema_handler = None
    if args.use_ema:
        ema_handler = FSDP_EMA(transformer, args.ema_decay, rank)

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)

    main_print(f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}")
    main_print(f"--> model loaded (using pre-computed T5 embeddings)")

    transformer.train()

    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = FluxFillLatentDataset(
        args.data_json_path, 
        num_latent_t=args.num_latent_t,
        cfg_rate=args.cfg,
    )
    
    sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=flux_fill_latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    if rank <= 0:
        swanlab.init(
            project="flux-fill-grpo",
            config=vars(args),
        )

    total_batch_size = (
        world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    
    # Calculate total training steps
    steps_per_epoch = len(train_dataloader)
    total_steps = min(args.max_train_steps, args.num_epochs * steps_per_epoch)
    
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(f"  Total train batch size = {total_batch_size}")
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps = {total_steps}")

    step_times = deque(maxlen=100)
    global_step = init_steps

    for epoch in range(args.num_epochs):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        for step, batch_data in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break
            
            start_time = time.time()
            
            # Check if we need to save checkpoint (not at step 0, and not the final step)
            is_final_step = (global_step + 1) >= args.max_train_steps
            if global_step > 0 and global_step % args.checkpointing_steps == 0 and not is_final_step:
                save_checkpoint(transformer, rank, args.output_dir, global_step, epoch)
                if args.use_ema:
                    save_ema_checkpoint(ema_handler, rank, args.output_dir, global_step, epoch, dict(transformer.config))
                dist.barrier()
            
            loss, grad_norm, reward_mean = train_one_step(
                args,
                device,
                transformer,
                vae,
                reward_model,
                optimizer,
                lr_scheduler,
                batch_data,
                args.max_grad_norm,
                ema_handler,
            )

            if args.use_ema and ema_handler:
                ema_handler.update(transformer)

            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
            
            # Calculate ETA
            remaining_steps = total_steps - global_step - 1
            eta_seconds = remaining_steps * avg_step_time
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60

            # Print progress info
            if local_rank == 0:
                eta_str = f"{eta_hours:.1f}h" if eta_hours >= 1 else f"{eta_minutes:.1f}m"
                grad_norm_str = f"{grad_norm:.4f}" if isinstance(grad_norm, float) else str(grad_norm)
                print(f"\n[Step {global_step+1}/{total_steps}] loss={loss:.4f} | reward={reward_mean:.4f} | grad_norm={grad_norm_str} | step_time={step_time:.2f}s | ETA: {eta_str}")
            
            if rank <= 0:
                swanlab.log({
                    "train_loss": loss,
                    "reward_mean": reward_mean,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                    "grad_norm": grad_norm,
                    "epoch": epoch,
                }, step=global_step)
            
            global_step += 1
        
        # Check if we've reached max steps
        if global_step >= args.max_train_steps:
            break

    # Save final model after training completes
    if args.final_model_dir:
        save_final_model(transformer, rank, args.final_model_dir)
        if args.use_ema and ema_handler:
            # Save EMA version as well
            ema_final_dir = args.final_model_dir + "_ema"
            save_ema_checkpoint(ema_handler, rank, ema_final_dir, global_step, epoch, dict(transformer.config))
        dist.barrier()
        main_print(f"--> Training completed. Final model saved to {args.final_model_dir}")
    else:
        main_print("--> Training completed. No final_model_dir specified, skipping final model save.")

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()
    
    if rank <= 0:
        swanlab.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_latent_t", type=int, default=1)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs/fill_grpo")
    parser.add_argument("--final_model_dir", type=str, default=None,
                        help="Directory to save the final trained model. If not specified, final model will not be saved separately.")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--use_cpu_offload", action="store_true")
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--train_sp_batch_size", type=int, default=1)
    parser.add_argument("--fsdp_sharding_startegy", default="full")
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--master_weight_type", type=str, default="fp32")
    parser.add_argument("--h", type=int, default=512)
    parser.add_argument("--w", type=int, default=512)
    parser.add_argument("--sampling_steps", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--use_group", action="store_true")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--init_same_noise", action="store_true")
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--timestep_fraction", type=float, default=0.6)
    parser.add_argument("--clip_range", type=float, default=1e-4)
    parser.add_argument("--adv_clip_max", type=float, default=5.0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--reward_type", type=str, default="inpainting",
                        choices=["clip", "clip_hps", "inpainting"])
    
    args = parser.parse_args()
    main(args)
