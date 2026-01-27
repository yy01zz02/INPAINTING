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
    config.num_epochs = 100
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
    config.checkpoint_dir = "./checkpoints/sd_inpainting_sft"
    # final model directory
    config.final_model_dir = "./final_models/sd_inpainting_sft"

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load
    pretrained.model = "runwayml/stable-diffusion-inpainting"
    # revision of the model to load
    pretrained.revision = "main"

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU) for training
    train.batch_size = 2
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
    # use classifier-free guidance during training
    train.cfg = False

    return config
