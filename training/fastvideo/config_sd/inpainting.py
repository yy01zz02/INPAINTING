import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = ""
    config.seed = 42
    config.logdir = "logs"
    config.num_epochs = 300
    config.save_freq = 20
    config.num_checkpoint_limit = 5
    config.mixed_precision = "bf16"
    config.allow_tf32 = True
    config.resume_from = ""
    config.use_lora = False

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "./data/stable-diffusion-inpainting"  # SD 1.5 inpainting model
    pretrained.revision = "main"

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    sample.num_steps = 50
    sample.eta = 1.0
    sample.guidance_scale = 7.5
    sample.batch_size = 1
    sample.num_batches_per_epoch = 2

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 1
    train.use_8bit_adam = False
    train.learning_rate = 1e-5
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    train.gradient_accumulation_steps = 4
    train.max_grad_norm = 1.0
    train.num_inner_epochs = 1
    train.cfg = True
    train.adv_clip_max = 5.0
    train.clip_range = 1e-4
    train.timestep_fraction = 0.6

    ###### Data ######
    config.data_json_path = "./data/train_metadata.jsonl"

    ###### Reward Function ######
    # Available types: 'clip', 'clip_hps', 'inpainting'
    # 'inpainting' uses API to get boundary_score, hps_score, clip_score, mask_ratio
    # and computes group-normalized reward
    # To change reward API endpoints, modify fastvideo/rewards/api_rewards.py
    config.reward_type = "inpainting"

    ###### GRPO Settings ######
    config.num_generations = 4
    config.max_train_steps = 10000  # Maximum training steps

    ###### Save Paths ######
    config.checkpoint_dir = "./checkpoints"  # Directory for intermediate checkpoints
    config.final_model_dir = ""  # Directory for final trained model (empty string = skip final save)

    return config
