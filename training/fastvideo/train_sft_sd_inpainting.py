# Supervised Fine-Tuning (SFT) script for Stable Diffusion 1.5 Inpainting model.
# Full fine-tuning without FSDP sharding (SD1.5 fits in single GPU).
# Uses SwanLab for logging.

from collections import defaultdict
import os
import datetime
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
import numpy as np
import torch
import torch.nn.functional as F
import swanlab
from functools import partial
import tqdm
from PIL import Image
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from safetensors.torch import save_file
from fastvideo.dataset.latent_sd_inpainting_rl_datasets import SDInpaintingDataset, sd_inpainting_collate_function

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "fastvideo/config_sd/sft.py", "Training configuration.")


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with=None,
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    
    # Initialize SwanLab
    if accelerator.is_main_process:
        swanlab.init(
            project="sd-inpainting-sft",
            config=config.to_dict(),
        )
    
    logger = get_logger(__name__)
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    # Load inpainting pipeline
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        config.pretrained.model if hasattr(config.pretrained, 'model') else "./data/stable-diffusion-inpainting",
        torch_dtype=torch.float32,
    )
    
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.safety_checker = None
    
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    
    unet = pipeline.unet
    unet.to(accelerator.device)
    
    # Enable full fine-tuning
    unet.requires_grad_(True)
    unet_trainable_parameters = list(unet.parameters())

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # Generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    autocast = accelerator.autocast

    unet, optimizer = accelerator.prepare(unet, optimizer)

    # Load inpainting dataset
    dataset = SDInpaintingDataset(
        config.data_json_path if hasattr(config, 'data_json_path') else "./data/train_metadata.jsonl",
        image_size=512,
    )

    dataset_size = len(dataset)
    steps_per_epoch = dataset_size // (config.train.batch_size * accelerator.num_processes)

    logger.info("***** Running SFT Training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Dataset size = {dataset_size}")
    logger.info(f"  Steps per epoch = {steps_per_epoch}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Max train steps = {config.max_train_steps}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
        rank=dist.get_rank() if dist.is_initialized() else 0,
        seed=123543,
        shuffle=True
    )

    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        sampler=sampler,
        collate_fn=sd_inpainting_collate_function,
        pin_memory=True,
        drop_last=True
    )

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    global_step = 0
    step_times = []
    
    for epoch in range(first_epoch, config.num_epochs):
        logger.info(f"Starting epoch {epoch}")
        
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        
        for step, batch_data in enumerate(loader):
            step_start_time = time.time()
            
            if global_step >= config.max_train_steps:
                break

            prompts = batch_data['prompts']
            images_pil = batch_data['images_pil']
            masks_pil = batch_data['masks_pil']
            
            #################### SFT TRAINING ####################
            unet.train()
            
            info = defaultdict(list)
            
            # Process each sample in batch
            batch_loss = 0.0
            valid_samples = 0
            
            for i in range(len(prompts)):
                current_prompt = prompts[i]
                current_image = images_pil[i]
                current_mask = masks_pil[i]
                
                # Encode prompt
                prompt_ids = pipeline.tokenizer(
                    [current_prompt],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length
                ).input_ids.to(accelerator.device)
                prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
                
                # Preprocess mask and image
                init_image = pipeline.image_processor.preprocess(
                    current_image, height=512, width=512
                )
                init_image = init_image.to(device=accelerator.device, dtype=inference_dtype)
                
                mask_condition = pipeline.mask_processor.preprocess(
                    current_mask, height=512, width=512
                )
                mask_condition = mask_condition.to(device=accelerator.device, dtype=inference_dtype)
                
                # Create masked image (mask out regions where mask > 0.5)
                masked_image = init_image * (mask_condition < 0.5)
                
                # Encode source image to latent space (this is our target)
                with torch.no_grad():
                    target_latents = pipeline.vae.encode(init_image).latent_dist.sample()
                    target_latents = target_latents * pipeline.vae.config.scaling_factor
                
                # Prepare mask and masked image latents
                mask, masked_image_latents = pipeline.prepare_mask_latents(
                    mask_condition,
                    masked_image,
                    1,  # batch_size
                    512,
                    512,
                    prompt_embeds.dtype,
                    accelerator.device,
                    None,  # generator
                    config.train.cfg if hasattr(config.train, 'cfg') else False,  # do_classifier_free_guidance
                )
                
                # Sample noise and timestep
                noise = torch.randn_like(target_latents)
                
                # Sample timestep using logit-normal distribution
                bsz = target_latents.shape[0]
                u = compute_density_for_timestep_sampling(
                    weighting_scheme='logit_normal',
                    batch_size=bsz,
                    logit_mean=0,
                    logit_std=1,
                    mode_scale=1.29,
                )
                indices = (u * pipeline.scheduler.config.num_train_timesteps).long()
                timesteps = pipeline.scheduler.timesteps[indices].to(device=accelerator.device)
                
                # Add noise to latents
                alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device=accelerator.device, dtype=target_latents.dtype)
                sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
                sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                while len(sqrt_alpha_prod.shape) < len(target_latents.shape):
                    sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                
                sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                while len(sqrt_one_minus_alpha_prod.shape) < len(target_latents.shape):
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                
                noisy_latents = sqrt_alpha_prod * target_latents + sqrt_one_minus_alpha_prod * noise
                
                # Prepare inpainting input (concatenate noisy latents with mask and masked image)
                if config.train.cfg if hasattr(config.train, 'cfg') else False:
                    # With classifier-free guidance, mask and masked_image_latents are doubled
                    inpaint_input = torch.cat([noisy_latents, mask[:1], masked_image_latents[:1]], dim=1)
                else:
                    inpaint_input = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)
                
                with accelerator.accumulate(unet):
                    with autocast():
                        # Predict noise
                        model_pred = unet(
                            inpaint_input,
                            timesteps,
                            encoder_hidden_states=prompt_embeds,
                            return_dict=False,
                        )[0]
                        
                        # MSE loss between predicted and actual noise
                        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                        
                        batch_loss += loss
                        valid_samples += 1
                        
                        info["loss"].append(loss.detach())
            
            if valid_samples > 0:
                avg_loss = batch_loss / valid_samples
                
                accelerator.backward(avg_loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                info_log = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                info_log = accelerator.reduce(info_log, reduction="mean")
                info_log.update({"epoch": epoch})
                
                if accelerator.is_main_process:
                    swanlab.log({
                        "loss": info_log["loss"].item(),
                    }, step=global_step)
                
                global_step += 1

            # Save checkpoint
            is_final_step = (global_step + 1) >= config.max_train_steps
            if step != 0 and global_step % config.save_freq == 0 and not is_final_step:
                if accelerator.is_main_process:
                    base_checkpoint_dir = config.checkpoint_dir if hasattr(config, 'checkpoint_dir') else "./checkpoints"
                    checkpoint_dir = os.path.join(base_checkpoint_dir, f"checkpoint_step_{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    unet_safetensors_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors")
                    unwrapped_unet = accelerator.unwrap_model(pipeline.unet)
                    model_state_dict = unwrapped_unet.state_dict()
                    save_file(model_state_dict, unet_safetensors_path)
                    
                    accelerator.print(f"Checkpoint saved to {checkpoint_dir}")

                if dist.is_initialized():
                    dist.barrier()
            
            # Calculate step time and ETA
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            if len(step_times) > 100:
                step_times = step_times[-100:]
            avg_step_time = sum(step_times) / len(step_times)
            remaining_steps = config.max_train_steps - global_step
            eta_seconds = remaining_steps * avg_step_time
            eta_hours = eta_seconds / 3600
            
            if accelerator.is_local_main_process:
                eta_str = f"{eta_hours:.1f}h" if eta_hours >= 1 else f"{eta_seconds/60:.1f}m"
                # Get current loss from info dict directly
                current_loss = torch.mean(torch.stack(info["loss"])).item() if len(info["loss"]) > 0 else 0.0
                print(f"\n[Step {global_step}/{config.max_train_steps}] epoch={epoch} | loss={current_loss:.6f} | step_time={step_time:.2f}s | ETA: {eta_str}")
        
        if global_step >= config.max_train_steps:
            break
    
    # Save final model
    if hasattr(config, 'final_model_dir') and config.final_model_dir and config.final_model_dir.strip():
        if accelerator.is_main_process:
            final_dir = config.final_model_dir
            os.makedirs(final_dir, exist_ok=True)
            
            unwrapped_unet = accelerator.unwrap_model(pipeline.unet)
            
            try:
                model_state_dict = unwrapped_unet.state_dict()
                unet_safetensors_path = os.path.join(final_dir, "diffusion_pytorch_model.safetensors")
                save_file(model_state_dict, unet_safetensors_path)
                
                accelerator.print(f"Training completed. Final model saved to {final_dir}")
            except Exception as e:
                accelerator.print(f"Error saving final model: {e}")
        
        if dist.is_initialized():
            dist.barrier()
    else:
        accelerator.print("Training completed. No final_model_dir specified, skipping final model save.")
    
    # Finish SwanLab
    if accelerator.is_main_process:
        swanlab.finish()


if __name__ == "__main__":
    app.run(main)
