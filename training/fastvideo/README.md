# Training Module

Core training module for GRPO and DPO algorithms on inpainting models.

## Directory Structure

```
fastvideo/
├── train_grpo_flux_fill.py       # Flux Fill GRPO training
├── train_dpo_flux_fill.py        # Flux Fill Online DPO training
├── train_sft_flux_fill.py        # Flux Fill SFT baseline
├── train_grpo_sd_inpainting.py   # SD1.5 Inpainting GRPO training
├── train_dpo_sd_inpainting.py    # SD1.5 Inpainting Online DPO training
├── train_sft_sd_inpainting.py    # SD1.5 Inpainting SFT baseline
├── config_sd/                    # Configuration files for SD1.5
├── data_preprocess/              # Data preprocessing scripts
├── dataset/                      # Dataset implementations
├── models/                       # Model architectures
├── rewards/                      # Reward functions
└── utils/                        # Utility functions
```

## Training Scripts

### Flux Fill

| Script | Algorithm | Features |
|--------|-----------|----------|
| `train_grpo_flux_fill.py` | GRPO | FSDP, EMA, gradient checkpointing |
| `train_dpo_flux_fill.py` | Online DPO | FSDP, reference model updates |
| `train_sft_flux_fill.py` | SFT | FSDP, supervised baseline |

### SD1.5 Inpainting

| Script | Algorithm | Features |
|--------|-----------|----------|
| `train_grpo_sd_inpainting.py` | GRPO | Accelerate, EMA weights |
| `train_dpo_sd_inpainting.py` | Online DPO | Accelerate, reference updates |
| `train_sft_sd_inpainting.py` | SFT | Accelerate, supervised baseline |

## Modules

### dataset/
- `latent_flux_fill_rl_datasets.py` - Dataset for Flux Fill with pre-computed T5 embeddings
- `latent_sd_inpainting_rl_datasets.py` - Dataset for SD1.5 inpainting

### models/
- `flux_hf/` - Flux model architecture
- `stable_diffusion/` - SD1.5 inpainting pipeline with log probability

### rewards/
- `api_rewards.py` - HTTP client for reward model servers
- `reward.py` - Reward computation utilities

### utils/
- `fsdp_util.py` - FSDP configuration
- `checkpoint.py` - Checkpoint saving/loading
- `parallel_states.py` - Distributed training states
