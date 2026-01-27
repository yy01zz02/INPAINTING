# Training script for Stable Diffusion 1.5 Inpainting model with GRPO.
# Full fine-tuning without FSDP sharding (SD1.5 fits in single GPU).
# Uses SwanLab for logging and CLIP + HPS API rewards.

from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, UNet2DConditionModel
import numpy as np
from fastvideo.models.stable_diffusion.pipeline_with_logprob import pipeline_with_logprob
from fastvideo.models.stable_diffusion.ddim_with_logprob import ddim_step_with_logprob
import torch
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
import torch.nn.functional as F

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
    # Use pipeline's image_processor and mask_processor
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
    
    # Return mask and masked_image_latents for training (only need one copy, not doubled for CFG)
    # mask shape: [2, 1, H, W] -> [1, 1, H, W]
    # masked_image_latents shape: [2, 4, H, W] -> [1, 4, H, W]
    return image, latents, all_latents, all_log_probs, mask[:1], masked_image_latents[:1]


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "fastvideo/config_sd/base.py", "Training configuration.")


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

    # For SD (diffusion model with SDE), train on all timesteps
    # timestep_fraction is only needed for flow models (ODE -> SDE conversion)
    # Subtract 1 because latents array has num_steps entries (excluding the final one)
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
        log_with=None,  # Using SwanLab directly instead of accelerator trackers
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    
    # Initialize SwanLab instead of WandB trackers
    if accelerator.is_main_process:
        swanlab.init(
            project="sd-inpainting-grpo",
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
    pipeline.unet.requires_grad_(True)
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

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()
    
    def gather_tensor(tensor):
        if not dist.is_initialized():
            return tensor
        world_size = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        return torch.cat(gathered_tensors, dim=0)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam."
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
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
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    autocast = accelerator.autocast

    unet, optimizer = accelerator.prepare(unet, optimizer)

    executor = futures.ThreadPoolExecutor(max_workers=2)

    # samples_per_step: 每个 dataloader step 产生的总样本数
    # = 每GPU采样batch * GPU数 * 每样本生成次数
    samples_per_step = (
        config.sample.batch_size
        * accelerator.num_processes
        * config.num_generations
    )
    # total_train_batch_size: 每次梯度更新消耗的样本数
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    # 断言：确保采样的样本能被均匀分成训练批次
    assert config.sample.batch_size >= config.train.batch_size, \
        "sample.batch_size must be >= train.batch_size"
    assert config.sample.batch_size % config.train.batch_size == 0, \
        "sample.batch_size must be divisible by train.batch_size"
    assert samples_per_step % total_train_batch_size == 0, \
        "samples_per_step must be divisible by total_train_batch_size"

    # Load inpainting dataset
    dataset = SDInpaintingDataset(
        config.data_json_path if hasattr(config, 'data_json_path') else "./data/train_metadata.jsonl",
        image_size=512,
    )

    # 计算每个 epoch 的实际步数（基于数据集大小）
    dataset_size = len(dataset)
    steps_per_epoch = dataset_size // (config.sample.batch_size * accelerator.num_processes)

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Dataset size = {dataset_size}")
    logger.info(f"  Steps per epoch = {steps_per_epoch}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Samples per step = {samples_per_step}")
    logger.info(f"  Total train batch size = {total_train_batch_size}")
    logger.info(f"  Gradient updates per step = {samples_per_step // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    logger.info(f"  Max train steps = {config.max_train_steps}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size() if dist.is_initialized() else 1,
        rank=torch.distributed.get_rank() if dist.is_initialized() else 0,
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
    step_times = []  # Track step times for ETA calculation
    
    for epoch in range(first_epoch, config.num_epochs):
        logger.info(f"Starting epoch {epoch}")
        
        # Set epoch for distributed sampler
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        
        for step, batch_data in enumerate(loader):
            step_start_time = time.time()
            prompts = batch_data['prompts']
            images_pil = batch_data['images_pil']  # PIL images for pipeline
            masks_pil = batch_data['masks_pil']    # PIL masks for pipeline
            
            #################### SAMPLING ####################
            pipeline.unet.eval()

            if global_step >= config.max_train_steps:
                break

            # Expand prompts for multiple generations
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
            all_raw_scores = []  # Store raw scores for inpainting reward
            all_prompts_embed = []
            all_masks = []
            all_masked_image_latents = []
            all_decoded_images = []  # Store decoded images for logging

            # Process one sample at a time for inpainting
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

                # Process single generated image
                image = gen_images[0]
                if isinstance(image, torch.Tensor):
                    pil = Image.fromarray(
                        (image.to(torch.float32).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                else:
                    pil = image
                pil = pil.resize((512, 512))
                
                image_path = os.path.join(
                    "./images", 
                    f"inpaint-{epoch}-{step}-{i}-rank-{dist.get_rank() if dist.is_initialized() else 0}.jpg"
                )
                os.makedirs("./images", exist_ok=True)
                pil.save(image_path)
                
                # Calculate raw reward scores
                if reward_type == 'inpainting':
                    raw_score = reward_model.compute_reward_raw(pil, current_mask, current_prompt)
                    all_raw_scores.append(raw_score)
                else:
                    # Fallback for non-inpainting reward types
                    score = reward_model.compute_reward(pil, current_prompt)
                    all_raw_scores.append({"reward": score})
                
                all_decoded_images.append(pil)

                latents = torch.stack(latents, dim=1).detach()
                log_probs = torch.stack(log_probs, dim=1).detach()

                all_latents.append(latents)
                all_log_probs.append(log_probs)
                all_prompts_embed.append(prompt_embeds)
                all_masks.append(mask_latent.detach())
                all_masked_image_latents.append(masked_img_latent.detach())

                torch.cuda.empty_cache()

            # Compute final rewards with group normalization
            if reward_type == 'inpainting' and len(all_raw_scores) > 0 and 'boundary_score' in all_raw_scores[0]:
                all_rewards = reward_model.compute_group_normalized_rewards(
                    all_raw_scores, config.num_generations
                )
            else:
                # Fallback: extract simple rewards
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
            
            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.batch_size * config.num_generations, 1
            )

            samples = {
                "prompt_embeds": all_prompts_embed,
                "timesteps": timesteps[:, :-1],
                "latents": all_latents[:, :-1],
                "next_latents": all_latents[:, 1:],
                "log_probs": all_log_probs,
                "rewards": all_rewards,
                "masks": all_masks,
                "masked_image_latents": all_masked_image_latents,
            }

            all_rewards_world = gather_tensor(all_rewards)

            # Log to SwanLab
            if accelerator.is_main_process:
                swanlab.log({
                    "reward_mean": all_rewards_world.mean().item(),
                    "reward_std": all_rewards_world.std().item(),
                    "epoch": epoch,
                    "step": step,
                }, step=global_step)

            if dist.is_initialized() and dist.get_rank() == 0:
                print("gathered_reward", all_rewards_world)
                with open('./reward.txt', 'a') as f:
                    f.write(f"{all_rewards_world.mean().item()}\n")

            # Compute advantages
            n = len(samples["rewards"]) // config.num_generations
            advantages = torch.zeros_like(samples["rewards"])

            for i in range(n):
                start_idx = i * config.num_generations
                end_idx = (i + 1) * config.num_generations
                group_rewards = samples["rewards"][start_idx:end_idx]
                group_mean = group_rewards.mean()
                group_std = group_rewards.std() + 1e-8
                advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
            
            samples["advantages"] = advantages
            samples["final_advantages"] = advantages

            total_batch_size_samples, num_timesteps = samples["timesteps"].shape

            #################### TRAINING ####################
            for inner_epoch in range(config.train.num_inner_epochs):
                # Shuffle along time dimension
                perms = torch.stack([
                    torch.randperm(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size_samples)
                ])
                
                for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                    samples[key] = samples[key][
                        torch.arange(total_batch_size_samples, device=accelerator.device)[:, None],
                        perms,
                    ]

                # Rebatch for training
                samples_batched = {
                    k: v.reshape(-1, config.train.batch_size, *v.shape[1:])
                    for k, v in samples.items()
                }
                samples_batched = [
                    dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
                ]

                pipeline.unet.train()
                info = defaultdict(list)
                
                for i, sample in tqdm(
                    list(enumerate(samples_batched)),
                    desc=f"Epoch {epoch}.{inner_epoch}: training",
                    position=0,
                    disable=not accelerator.is_local_main_process,
                ):
                    if config.train.cfg:
                        embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
                    else:
                        embeds = sample["prompt_embeds"]

                    for j in tqdm(
                        range(num_train_timesteps),
                        desc="Timestep",
                        position=1,
                        leave=False,
                        disable=not accelerator.is_local_main_process,
                    ):
                        with accelerator.accumulate(unet):
                            with autocast():
                                # For inpainting, concatenate latents with mask and masked_image_latents
                                # UNet expects 9 channels: 4 (latent) + 1 (mask) + 4 (masked_image_latent)
                                current_latents = sample["latents"][:, j]
                                current_masks = sample["masks"]
                                current_masked_image_latents = sample["masked_image_latents"]
                                
                                # Concatenate: [batch, 4, H, W] + [batch, 1, H, W] + [batch, 4, H, W] -> [batch, 9, H, W]
                                latent_model_input = torch.cat(
                                    [current_latents, current_masks, current_masked_image_latents], dim=1
                                )
                                
                                if config.train.cfg:
                                    noise_pred = unet(
                                        torch.cat([latent_model_input] * 2),
                                        torch.cat([sample["timesteps"][:, j]] * 2),
                                        embeds,
                                    ).sample
                                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                    noise_pred = (
                                        noise_pred_uncond
                                        + config.sample.guidance_scale
                                        * (noise_pred_text - noise_pred_uncond)
                                    )
                                else:
                                    noise_pred = unet(
                                        latent_model_input,
                                        sample["timesteps"][:, j],
                                        embeds,
                                    ).sample
                                
                                _, log_prob = ddim_step_with_logprob(
                                    pipeline.scheduler,
                                    noise_pred,
                                    sample["timesteps"][:, j],
                                    sample["latents"][:, j],
                                    eta=config.sample.eta,
                                    prev_sample=sample["next_latents"][:, j],
                                )

                            # GRPO loss
                            advantages = torch.clamp(
                                sample["final_advantages"],
                                -config.train.adv_clip_max,
                                config.train.adv_clip_max,
                            )
                            ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                            unclipped_loss = -advantages * ratio
                            clipped_loss = -advantages * torch.clamp(
                                ratio,
                                1.0 - config.train.clip_range,
                                1.0 + config.train.clip_range,
                            )
                            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                            info["approx_kl"].append(
                                0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                            )
                            info["clipfrac"].append(
                                torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float())
                            )
                            info["loss"].append(loss)

                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()

                        if accelerator.sync_gradients:
                            assert (j == num_train_timesteps - 1) and (i + 1) % config.train.gradient_accumulation_steps == 0
                            info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                            info = accelerator.reduce(info, reduction="mean")
                            info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                            
                            # Log to SwanLab
                            if accelerator.is_main_process:
                                swanlab.log({
                                    "loss": info["loss"].item(),
                                    "approx_kl": info["approx_kl"].item(),
                                    "clipfrac": info["clipfrac"].item(),
                                }, step=global_step)
                            
                            global_step += 1
                            info = defaultdict(list)
                    
                    if dist.is_initialized() and dist.get_rank() % 8 == 0:
                        print("reward", sample["rewards"])
                        print("ratio", ratio)
                        print("final advantage", advantages)
                        print("final loss", loss)
                    
                    if dist.is_initialized():
                        dist.barrier()

                assert accelerator.sync_gradients

            # Save checkpoint (skip if it's the final step)
            is_final_step = (global_step + 1) >= config.max_train_steps
            if step != 0 and global_step % config.save_freq == 0 and not is_final_step:
                if accelerator.is_main_process:
                    base_checkpoint_dir = config.checkpoint_dir if hasattr(config, 'checkpoint_dir') else "./checkpoints"
                    checkpoint_dir = os.path.join(base_checkpoint_dir, f"checkpoint_step_{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    unet_safetensors_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors")

                    unwrapped_unet = accelerator.unwrap_model(pipeline.unet)

                    try:
                        model_state_dict = unwrapped_unet.state_dict()
                        save_file(model_state_dict, unet_safetensors_path)
                        # Save config as well
                        config_path = os.path.join(checkpoint_dir, "config.json")
                        config_dict = dict(unwrapped_unet.config)
                        with open(config_path, 'w') as f:
                            json.dump(config_dict, f, indent=4)
                        accelerator.print(f"Checkpoint saved to {checkpoint_dir}")
                    except Exception as e:
                        accelerator.print(f"Error saving checkpoint: {e}")

                if dist.is_initialized():
                    dist.barrier()
            
            # Calculate step time and ETA
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            if len(step_times) > 100:
                step_times = step_times[-100:]  # Keep last 100 for moving average
            avg_step_time = sum(step_times) / len(step_times)
            remaining_steps = config.max_train_steps - global_step
            eta_seconds = remaining_steps * avg_step_time
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
            
            # Print progress
            if accelerator.is_local_main_process:
                eta_str = f"{eta_hours:.1f}h" if eta_hours >= 1 else f"{eta_minutes:.1f}m"
                reward_mean = all_rewards.mean().item() if hasattr(all_rewards, 'mean') else 0
                print(f"\n[Step {global_step}/{config.max_train_steps}] epoch={epoch} | reward_mean={reward_mean:.4f} | step_time={step_time:.2f}s | ETA: {eta_str}")
        
        # Check max steps at end of epoch
        if global_step >= config.max_train_steps:
            break
    
    # Save final model after training completes
    if hasattr(config, 'final_model_dir') and config.final_model_dir and config.final_model_dir.strip():
        if accelerator.is_main_process:
            final_dir = config.final_model_dir
            os.makedirs(final_dir, exist_ok=True)
            
            unwrapped_unet = accelerator.unwrap_model(pipeline.unet)
            
            try:
                # Save final model weights
                model_state_dict = unwrapped_unet.state_dict()
                unet_safetensors_path = os.path.join(final_dir, "diffusion_pytorch_model.safetensors")
                save_file(model_state_dict, unet_safetensors_path)
                
                # Save config
                config_path = os.path.join(final_dir, "config.json")
                config_dict = dict(unwrapped_unet.config)
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=4)
                
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
