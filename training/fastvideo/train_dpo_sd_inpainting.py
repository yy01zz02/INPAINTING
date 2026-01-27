# Training script for Stable Diffusion 1.5 Inpainting model with Online DPO.
# Adapted from FlowGRPO's online DPO training approach.
# Uses SwanLab for logging and CLIP + HPS API rewards.

from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
from copy import deepcopy
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.training_utils import compute_density_for_timestep_sampling
import numpy as np
from fastvideo.models.stable_diffusion.pipeline_with_logprob import pipeline_with_logprob
from fastvideo.models.stable_diffusion.ddim_with_logprob import ddim_step_with_logprob
import torch
import torch.nn.functional as F
import swanlab
from functools import partial
import tqdm
from PIL import Image
import torch.distributed as dist
import json
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from safetensors.torch import save_file
from fastvideo.rewards import get_reward_fn
from fastvideo.dataset.latent_sd_inpainting_rl_datasets import SDInpaintingDataset, sd_inpainting_collate_function

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


def pipeline_inpainting_with_logprob(
    pipeline,
    prompt_embeds,
    negative_prompt_embeds,
    image,
    mask_image,
    num_inference_steps,
    guidance_scale,
    eta,
    output_type="pt",
    latents=None,
    height=512,
    width=512,
):
    """Modified pipeline for inpainting with log probability calculation."""
    device = pipeline._execution_device
    
    # Preprocess mask and image (convert PIL to tensor)
    init_image = pipeline.image_processor.preprocess(
        image, height=height, width=width
    )
    init_image = init_image.to(device=device, dtype=prompt_embeds.dtype)
    
    mask_condition = pipeline.mask_processor.preprocess(
        mask_image, height=height, width=width
    )
    mask_condition = mask_condition.to(device=device, dtype=prompt_embeds.dtype)
    
    # Create masked image (mask out regions where mask > 0.5)
    masked_image = init_image * (mask_condition < 0.5)
    
    # Prepare mask and masked image latents
    mask, masked_image_latents = pipeline.prepare_mask_latents(
        mask_condition,
        masked_image,
        1,  # batch_size
        height,
        width,
        prompt_embeds.dtype,
        device,
        None,  # generator
        True,  # do_classifier_free_guidance
    )
    
    # Calculate latent dimensions based on height/width
    latent_height = height // pipeline.vae_scale_factor
    latent_width = width // pipeline.vae_scale_factor
    
    # Prepare latents
    if latents is None:
        latents = torch.randn(
            (1, 4, latent_height, latent_width),
            device=device,
            dtype=prompt_embeds.dtype,
        )
    
    # Set timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    
    all_latents = [latents]
    all_log_probs = []
    
    # Prepare extra kwargs for scheduler step
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(None, eta)
    
    # Denoising loop
    for i, t in enumerate(timesteps):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
        
        # Concatenate mask and masked image latents
        latent_model_input = torch.cat(
            [latent_model_input, mask, masked_image_latents], dim=1
        )
        
        # Predict noise residual
        noise_pred = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
            return_dict=False,
        )[0]
        
        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute previous noisy sample and log probability
        prev_sample, log_prob = ddim_step_with_logprob(
            pipeline.scheduler,
            noise_pred,
            t,
            latents,
            eta=eta,
            prev_sample=None,
        )
        
        latents = prev_sample
        all_latents.append(latents)
        all_log_probs.append(log_prob)
    
    # Decode latents
    image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    
    # Post-process
    do_denormalize = [True] * image.shape[0]
    image = pipeline.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
    
    return image, latents, all_latents, all_log_probs, mask[:1], masked_image_latents[:1]


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "fastvideo/config_sd/dpo.py", "Training configuration.")


class EMAModuleWrapper:
    """Simple EMA wrapper for model parameters."""
    
    def __init__(self, parameters, decay=0.9, update_step_interval=8, device='cuda'):
        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device
        self.shadow_params = [p.clone().detach().to(device) for p in parameters]
        self.temp_params = None
        
    def step(self, parameters, step):
        if step % self.update_step_interval == 0:
            with torch.no_grad():
                for shadow, param in zip(self.shadow_params, parameters):
                    shadow.mul_(self.decay).add_(param.data, alpha=1-self.decay)
    
    def copy_ema_to(self, parameters, store_temp=False):
        if store_temp:
            self.temp_params = [p.clone() for p in parameters]
        with torch.no_grad():
            for param, shadow in zip(parameters, self.shadow_params):
                param.data.copy_(shadow)
    
    def copy_temp_to(self, parameters):
        if self.temp_params is not None:
            with torch.no_grad():
                for param, temp in zip(parameters, self.temp_params):
                    param.data.copy_(temp)
            self.temp_params = None


def calculate_zero_std_ratio(prompts, rewards):
    """Calculate the proportion of unique prompts whose reward standard deviation is zero."""
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    grouped_rewards = rewards[np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()


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

    num_train_timesteps = config.sample.num_steps - 1

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    # Initialize reward model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward_type = config.reward_type if hasattr(config, 'reward_type') else 'clip_hps'
    reward_model = get_reward_fn(reward_type, device)
    print(f"Using reward model: {reward_type}")
    
    accelerator = Accelerator(
        log_with=None,
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    
    # Initialize SwanLab
    if accelerator.is_main_process:
        swanlab.init(
            project="sd-inpainting-online-dpo",
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
    
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    
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
    
    # Enable full fine-tuning and create separate reference model
    unet.requires_grad_(True)
    unet_ref = deepcopy(unet)
    unet_ref.requires_grad_(False)
    unet_ref.to(accelerator.device)
    
    # Collect all trainable parameters for full fine-tuning
    unet_trainable_parameters = list(unet.parameters())
    
    # Initialize EMA
    ema = None
    if hasattr(config.train, 'ema') and config.train.ema:
        ema = EMAModuleWrapper(
            unet_trainable_parameters, 
            decay=0.9, 
            update_step_interval=8, 
            device=accelerator.device
        )

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
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    autocast = accelerator.autocast

    unet, optimizer = accelerator.prepare(unet, optimizer)

    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Load inpainting dataset
    dataset = SDInpaintingDataset(
        config.data_json_path if hasattr(config, 'data_json_path') else "./data/train_metadata.jsonl",
        image_size=512,
    )

    dataset_size = len(dataset)
    steps_per_epoch = dataset_size // (config.sample.batch_size * accelerator.num_processes)

    logger.info("***** Running Online DPO Training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Dataset size = {dataset_size}")
    logger.info(f"  Steps per epoch = {steps_per_epoch}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Max train steps = {config.max_train_steps}")
    logger.info(f"  DPO beta = {config.train.beta}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
        rank=dist.get_rank() if dist.is_initialized() else 0,
        seed=123543,
        shuffle=True
    )

    loader = DataLoader(
        dataset,
        batch_size=config.sample.batch_size,
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
            prompts = batch_data['prompts']
            images_pil = batch_data['images_pil']
            masks_pil = batch_data['masks_pil']
            
            #################### SAMPLING ####################
            pipeline.unet.eval()

            if global_step >= config.max_train_steps:
                break

            # Update reference model periodically
            if global_step > 0 and global_step % config.train.ref_update_step == 0:
                unet_ref.load_state_dict(accelerator.unwrap_model(unet).state_dict())

            # Expand prompts for multiple generations (need at least 2 for DPO pairs)
            expanded_prompts = []
            expanded_images = []
            expanded_masks = []
            for p, img, msk in zip(prompts, images_pil, masks_pil):
                for _ in range(config.num_generations):
                    expanded_prompts.append(p)
                    expanded_images.append(img)
                    expanded_masks.append(msk)

            all_latents = []
            all_log_probs = []
            all_raw_scores = []
            all_prompts_embed = []
            all_masks = []
            all_masked_image_latents = []
            all_decoded_images = []

            # Sample from reference model
            for i in tqdm(
                range(len(expanded_prompts)),
                desc=f"Sampling [{step}]",
                leave=False,
                disable=not accelerator.is_local_main_process,
            ):
                current_prompt = expanded_prompts[i]
                current_image = expanded_images[i]
                current_mask = expanded_masks[i]

                prompt_ids = pipeline.tokenizer(
                    [current_prompt],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length
                ).input_ids.to(accelerator.device)
                prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
                
                # Generate new noise for each new prompt, reuse for same prompt's generations
                if i % config.num_generations == 0:
                    input_latents = torch.randn(
                        (1, 4, 64, 64),
                        device=accelerator.device,
                        dtype=inference_dtype,
                    )

                with torch.no_grad():
                    with autocast():
                        gen_images, _, latents, log_probs, mask_latent, masked_img_latent = pipeline_inpainting_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds[0:1],
                            image=current_image,
                            mask_image=current_mask,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            eta=config.sample.eta,
                            output_type="pt",
                            latents=input_latents,
                            height=512,
                            width=512,
                        )

                # Process generated image
                image = gen_images[0]
                if isinstance(image, torch.Tensor):
                    pil = Image.fromarray(
                        (image.to(torch.float32).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                else:
                    pil = image
                pil = pil.resize((512, 512))
                
                # Calculate reward scores
                if reward_type == 'inpainting':
                    raw_score = reward_model.compute_reward_raw(pil, current_mask, current_prompt)
                    all_raw_scores.append(raw_score)
                else:
                    score = reward_model.compute_reward(pil, current_prompt)
                    all_raw_scores.append({"reward": score})
                
                all_decoded_images.append(pil)

                latents_tensor = torch.stack(latents, dim=1).detach()
                log_probs_tensor = torch.stack(log_probs, dim=1).detach()

                all_latents.append(latents_tensor)
                all_log_probs.append(log_probs_tensor)
                all_prompts_embed.append(prompt_embeds)
                all_masks.append(mask_latent.detach())
                all_masked_image_latents.append(masked_img_latent.detach())

                torch.cuda.empty_cache()

            # Compute rewards
            if reward_type == 'inpainting' and len(all_raw_scores) > 0 and 'boundary_score' in all_raw_scores[0]:
                all_rewards = reward_model.compute_group_normalized_rewards(
                    all_raw_scores, config.num_generations
                )
            else:
                all_rewards = torch.tensor(
                    [s.get("reward", 0.0) for s in all_raw_scores], 
                    device=accelerator.device, 
                    dtype=torch.float32
                )

            all_latents = torch.cat(all_latents, dim=0)
            all_log_probs = torch.cat(all_log_probs, dim=0)
            all_prompts_embed = torch.cat(all_prompts_embed, dim=0)
            all_masks = torch.cat(all_masks, dim=0)
            all_masked_image_latents = torch.cat(all_masked_image_latents, dim=0)

            # Gather rewards across processes
            all_rewards_world = gather_tensor(all_rewards)

            # Log to SwanLab
            if accelerator.is_main_process:
                swanlab.log({
                    "reward_mean": all_rewards_world.mean().item(),
                    "reward_std": all_rewards_world.std().item(),
                    "epoch": epoch,
                    "step": step,
                }, step=global_step)

            # Prepare DPO pairs based on rewards
            # Group by prompt and select winner/loser based on reward
            n_prompts = len(prompts)
            n_gens = config.num_generations
            
            # Calculate advantages for filtering
            advantages = torch.zeros_like(all_rewards)
            for i in range(n_prompts):
                start_idx = i * n_gens
                end_idx = (i + 1) * n_gens
                group_rewards = all_rewards[start_idx:end_idx]
                group_mean = group_rewards.mean()
                group_std = group_rewards.std() + 1e-8
                advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
            
            advantages = advantages.cpu().numpy()
            all_rewards_np = all_rewards.cpu().numpy()
            
            # Filter out samples with non-zero advantages
            non_zero_indices = np.where(advantages != 0)[0]
            
            if len(non_zero_indices) == 0:
                logger.warning("All advantages are zero, skipping this batch")
                continue
            
            # Group latents by prompt and create winner/loser pairs
            concat_latent_w = []
            concat_latent_l = []
            concat_prompt_embeds = []
            concat_masks = []
            concat_masked_image_latents = []
            
            for i in range(n_prompts):
                start_idx = i * n_gens
                end_idx = (i + 1) * n_gens
                group_rewards = all_rewards_np[start_idx:end_idx]
                group_advantages = advantages[start_idx:end_idx]
                
                # Skip if all rewards are the same
                if np.std(group_rewards) < 1e-6:
                    continue
                
                # Get winner (highest reward) and loser (lowest reward)
                winner_idx = start_idx + np.argmax(group_rewards)
                loser_idx = start_idx + np.argmin(group_rewards)
                
                # Only include if winner != loser
                if winner_idx != loser_idx:
                    concat_latent_w.append(all_latents[winner_idx:winner_idx+1])
                    concat_latent_l.append(all_latents[loser_idx:loser_idx+1])
                    concat_prompt_embeds.append(all_prompts_embed[winner_idx:winner_idx+1])
                    concat_masks.append(all_masks[winner_idx:winner_idx+1])
                    concat_masked_image_latents.append(all_masked_image_latents[winner_idx:winner_idx+1])
            
            if len(concat_latent_w) == 0:
                logger.warning("No valid DPO pairs found, skipping this batch")
                continue
            
            concat_latent_w = torch.cat(concat_latent_w, dim=0)
            concat_latent_l = torch.cat(concat_latent_l, dim=0)
            concat_prompt_embeds = torch.cat(concat_prompt_embeds, dim=0)
            concat_masks = torch.cat(concat_masks, dim=0)
            concat_masked_image_latents = torch.cat(concat_masked_image_latents, dim=0)
            
            # Stack winner and loser latents: [num_pairs, 2, T, C, H, W]
            dpo_latents = torch.stack([concat_latent_w, concat_latent_l], dim=1)
            
            samples = {
                "latents": dpo_latents,  # [num_pairs, 2, T, C, H, W]
                "prompt_embeds": concat_prompt_embeds,
                "masks": concat_masks,
                "masked_image_latents": concat_masked_image_latents,
            }

            num_dpo_pairs = dpo_latents.shape[0]

            #################### DPO TRAINING ####################
            pipeline.unet.train()
            
            info = defaultdict(list)
            
            for inner_epoch in range(config.train.num_inner_epochs):
                # Shuffle DPO pairs
                perm = torch.randperm(num_dpo_pairs, device=accelerator.device)
                
                for mini_batch_start in range(0, num_dpo_pairs, config.train.batch_size):
                    mini_batch_end = min(mini_batch_start + config.train.batch_size, num_dpo_pairs)
                    mini_batch_idx = perm[mini_batch_start:mini_batch_end].cpu()
                    
                    batch_latents = samples["latents"][mini_batch_idx]  # [B, 2, T, C, H, W]
                    batch_embeds = samples["prompt_embeds"][mini_batch_idx]
                    batch_masks = samples["masks"][mini_batch_idx]
                    batch_masked_image_latents = samples["masked_image_latents"][mini_batch_idx]
                    
                    bsz = batch_latents.shape[0]
                    
                    # Flatten winner and loser
                    model_input_w = batch_latents[:, 0, -1]  # Winner final latent [B, C, H, W]
                    model_input_l = batch_latents[:, 1, -1]  # Loser final latent [B, C, H, W]
                    model_input = torch.cat([model_input_w, model_input_l], dim=0)  # [2B, C, H, W]
                    
                    # Create noise (same noise for winner and loser as in Diffusion-DPO)
                    noise = torch.randn_like(model_input_w)
                    noise = torch.cat([noise, noise], dim=0)
                    
                    # Sample random timesteps
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme='logit_normal',
                        batch_size=bsz,
                        logit_mean=0,
                        logit_std=1,
                        mode_scale=1.29,
                    )
                    indices = (u * pipeline.scheduler.config.num_train_timesteps).long()
                    timesteps = pipeline.scheduler.timesteps[indices].to(device=model_input.device)
                    timesteps = torch.cat([timesteps, timesteps], dim=0)
                    
                    # Add noise to latents
                    alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device=model_input.device, dtype=model_input.dtype)
                    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                    while len(sqrt_alpha_prod.shape) < len(model_input.shape):
                        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                    
                    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                    while len(sqrt_one_minus_alpha_prod.shape) < len(model_input.shape):
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                    
                    noisy_model_input = sqrt_alpha_prod * model_input + sqrt_one_minus_alpha_prod * noise
                    
                    # Prepare inpainting input (concatenate with mask and masked image)
                    batch_masks_expanded = torch.cat([batch_masks, batch_masks], dim=0)
                    batch_masked_image_latents_expanded = torch.cat([batch_masked_image_latents, batch_masked_image_latents], dim=0)
                    
                    inpaint_input = torch.cat([noisy_model_input, batch_masks_expanded, batch_masked_image_latents_expanded], dim=1)
                    embeds_expanded = torch.cat([batch_embeds, batch_embeds], dim=0)
                    
                    with accelerator.accumulate(unet):
                        with autocast():
                            # Learner model prediction
                            model_pred = unet(
                                inpaint_input,
                                timesteps,
                                encoder_hidden_states=embeds_expanded,
                                return_dict=False,
                            )[0]
                            
                            # Reference model prediction
                            with torch.no_grad():
                                model_pred_ref = unet_ref(
                                    inpaint_input,
                                    timesteps,
                                    encoder_hidden_states=embeds_expanded,
                                    return_dict=False,
                                )[0]
                                model_pred_ref = model_pred_ref.detach()
                        
                        # Target is the noise
                        target = noise
                        
                        # Compute MSE errors
                        theta_mse = ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1).mean(dim=1)
                        ref_mse = ((model_pred_ref.float() - target.float()) ** 2).reshape(target.shape[0], -1).mean(dim=1)
                        
                        # Split into winner and loser
                        model_w_err = theta_mse[:bsz]
                        model_l_err = theta_mse[bsz:]
                        ref_w_err = ref_mse[:bsz]
                        ref_l_err = ref_mse[bsz:]
                        
                        # DPO loss
                        w_diff = model_w_err - ref_w_err
                        l_diff = model_l_err - ref_l_err
                        w_l_diff = w_diff - l_diff
                        
                        inside_term = -0.5 * config.train.beta * w_l_diff
                        loss = -F.logsigmoid(inside_term)
                        loss = torch.mean(loss)
                        
                        # Compute implicit accuracy
                        implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                        
                        info["loss"].append(loss)
                        info["model_w_err"].append(torch.mean(model_w_err))
                        info["model_l_err"].append(torch.mean(model_l_err))
                        info["ref_w_err"].append(torch.mean(ref_w_err))
                        info["ref_l_err"].append(torch.mean(ref_l_err))
                        info["w_diff"].append(torch.mean(w_diff))
                        info["l_diff"].append(torch.mean(l_diff))
                        info["w_l_diff"].append(torch.mean(w_l_diff))
                        info["inside_term"].append(torch.mean(inside_term))
                        info["implicit_acc"].append(implicit_acc)
                        
                        accelerator.backward(loss)
                        
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                        
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    if accelerator.sync_gradients:
                        info_log = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info_log = accelerator.reduce(info_log, reduction="mean")
                        info_log.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        
                        if accelerator.is_main_process:
                            swanlab.log({
                                "loss": info_log["loss"].item(),
                                "implicit_acc": info_log["implicit_acc"].item(),
                                "w_diff": info_log["w_diff"].item(),
                                "l_diff": info_log["l_diff"].item(),
                            }, step=global_step)
                        
                        info = defaultdict(list)
                        global_step += 1
                        
                        if ema is not None:
                            ema.step(unet_trainable_parameters, global_step)

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
                reward_mean = all_rewards.mean().item()
                print(f"\n[Step {global_step}/{config.max_train_steps}] epoch={epoch} | reward_mean={reward_mean:.4f} | step_time={step_time:.2f}s | ETA: {eta_str}")
        
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
