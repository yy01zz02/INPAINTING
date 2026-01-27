import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    # run name for logging and checkpoint saving
    config.run_name = ""
    # random seed for reproducibility
    config.seed = 42
    # top-level logging directory for checkpoint saving
    config.logdir = "logs"
    # number of epochs to train for
    config.num_epochs = 300
    # number of steps between saving model checkpoints
    config.save_freq = 100
    # number of checkpoints to keep before overwriting old ones
    config.num_checkpoint_limit = 5
    # mixed precision training: "fp16", "bf16", or "no"
    config.mixed_precision = "bf16"
    # allow tf32 on Ampere GPUs
    config.allow_tf32 = True
    # resume training from a checkpoint
    config.resume_from = ""
    # maximum training steps
    config.max_train_steps = 1000
    # data json path for inpainting dataset
    config.data_json_path = "./data/train_metadata.jsonl"
    # checkpoint directory
    config.checkpoint_dir = "./checkpoints/sd_inpainting_dpo"
    # final model directory
    config.final_model_dir = "./final_models/sd_inpainting_dpo"
    # reward type: 'clip', 'clip_hps', 'inpainting'
    config.reward_type = "inpainting"
    # number of generations per prompt for DPO pair selection
    config.num_generations = 4

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load
    pretrained.model = "runwayml/stable-diffusion-inpainting"
    # revision of the model to load
    pretrained.revision = "main"

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps
    sample.num_steps = 50
    # eta parameter for the DDIM sampler
    sample.eta = 1.0
    # classifier-free guidance weight (1.0 = no guidance)
    sample.guidance_scale = 7.5
    # batch size (per GPU) for sampling
    sample.batch_size = 2
    # number of batches to sample per epoch
    sample.num_batches_per_epoch = 2

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU) for training
    train.batch_size = 1
    # use 8bit Adam optimizer from bitsandbytes
    train.use_8bit_adam = False
    # learning rate
    train.learning_rate = 1e-5
    # Adam beta1
    train.adam_beta1 = 0.9
    # Adam beta2
    train.adam_beta2 = 0.999
    # Adam weight decay
    train.adam_weight_decay = 1e-4
    # Adam epsilon
    train.adam_epsilon = 1e-8
    # gradient accumulation steps
    train.gradient_accumulation_steps = 4
    # maximum gradient norm for clipping
    train.max_grad_norm = 1.0
    # number of inner epochs per outer epoch
    train.num_inner_epochs = 1
    # use classifier-free guidance during training
    train.cfg = True
    # clip advantages range
    train.adv_clip_max = 5.0
    # PPO clip range (unused in pure DPO but kept for compatibility)
    train.clip_range = 1e-4
    # fraction of timesteps to train on
    train.timestep_fraction = 1.0
    
    ###### DPO Specific ######
    # DPO beta parameter - controls preference optimization strength
    # Higher values = stronger preference signal, lower = more regularization
    # Typical range: 1000-10000 for diffusion DPO
    train.beta = 5000.0
    # reference model update frequency (steps)
    # Set to 0 to disable reference model updates (offline DPO)
    train.ref_update_step = 100
    # use EMA for model weights
    train.ema = False

    ###### Reward Function ######
    # reward function to use (legacy, use config.reward_type instead)
    config.reward_fn = "inpainting"

    return config
