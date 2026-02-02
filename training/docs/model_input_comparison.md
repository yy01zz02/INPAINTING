# Model Input Processing Comparison

This document describes the input processing for different models (Flux Fill, SD1.5 Inpainting) in the training framework.

## 目录

1. [输入类型概览](#1-输入类型概览)
2. [数据格式对比](#2-数据格式对比)
3. [数据集实现](#3-数据集实现)
4. [训练流程对比](#4-训练流程对比)
5. [采样过程对比](#5-采样过程对比)
6. [Transformer 输入构建](#6-transformer-输入构建)

---

## 1. 输入类型概览

| 模型 | 输入类型 | 任务描述 |
|------|----------|----------|
| **Flux Fill** | Prompt + 源图片 + Mask | 基于 mask 进行图像修复/补全 |
| **SD1.5 Inpainting** | Prompt + 源图片 + Mask | 基于 mask 进行图像修复/补全 |

### 详细输入说明

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Flux Fill (Inpainting)                                                  │
│ ├── prompt: "a red car"                                                 │
│ ├── source_image: 原始图片                                                │ 
│ ├── mask: 二值化掩码 (白色区域为需要修复的部分)                               │
│ └── 输出: 修复后的图片                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ SD1.5 Inpainting                                                        │
│ ├── prompt: "watercolor painting"                                       │
│ ├── source_image: 原始图片                                               │
│ ├── mask: 二值化掩码                                                      │
│ └── 输出: 修复后的图片                                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据格式对比

### 2.1 Flux Fill 数据格式

```json
{
  "caption": "a red car on the street",
  "prompt_embed_path": "fill_0.pt",
  "pooled_prompt_embeds_path": "fill_0.pt",
  "text_ids": "fill_0.pt",
  "masked_latents_path": "fill_0.pt",
  "mask_latents_path": "fill_0.pt"
}
```

**预计算的嵌入文件:**
- `prompt_embed/`: T5 嵌入
- `pooled_prompt_embeds/`: CLIP 池化输出
- `text_ids/`: 文本位置 ID
- `masked_latents/`: 遮罩图像潜在向量 `source * (1-mask)` -> `[16, H/8, W/8]`
- `mask_latents/`: 下采样的 mask `[1, H/8, W/8]`

### 2.2 SD1.5 Inpainting 数据格式

**原始 JSONL 格式:**
```json
{
  "prompt": "Calle De Portugal Acuarela S Papel 55x36 Cm Watercolor",
  "image": "images/train_000000.png",
  "mask": "masks/train_000000.png"
}
```

**运行时返回的数据:**
```python
{
    'prompt': str,           # 文本提示
    'image': Tensor,         # [3, H, W] 归一化图像
    'mask': Tensor,          # [1, H, W] 二值化掩码
    'masked_image': Tensor,  # [3, H, W] 遮罩后图像
    'image_pil': PIL.Image,  # 用于 pipeline 的 PIL 图像
    'mask_pil': PIL.Image,   # 用于 pipeline 的 PIL 掩码
    'image_path': str,       # 图片路径
    'mask_path': str,        # 掩码路径
}
```

---

## 3. 数据集实现

### 3.1 Flux Fill - FluxFillLatentDataset

```python
# fastvideo/dataset/latent_flux_fill_rl_datasets.py

class FluxFillLatentDataset(Dataset):
    def __init__(self, json_path, num_latent_t, cfg_rate):
        # 文本嵌入目录
        self.prompt_embed_dir = os.path.join(self.data_dir, "prompt_embed")
        self.pooled_prompt_embeds_dir = os.path.join(self.data_dir, "pooled_prompt_embeds")
        self.text_ids_dir = os.path.join(self.data_dir, "text_ids")
        # Inpainting 特有目录
        self.masked_latents_dir = os.path.join(self.data_dir, "masked_latents")
        self.mask_latents_dir = os.path.join(self.data_dir, "mask_latents")
        # ...

    def __getitem__(self, idx):
        # 加载文本嵌入
        prompt_embed = torch.load(...)
        pooled_prompt_embeds = torch.load(...)
        text_ids = torch.load(...)
        
        # 加载遮罩相关潜在向量 (Fill 特有)
        masked_latents = torch.load(...)  # source * (1-mask) 的潜在向量
        mask_latents = torch.load(...)     # 下采样的 mask
        
        return (prompt_embed, pooled_prompt_embeds, text_ids, 
                masked_latents, mask_latents, caption)
```

### 3.2 SD1.5 Inpainting - SDInpaintingDataset

```python
# fastvideo/dataset/latent_sd_inpainting_rl_datasets.py

class SDInpaintingDataset(Dataset):
    def __init__(self, jsonl_path, image_size=512, cfg_rate=0.0):
        # 不需要预计算，运行时加载
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        # ...

    def __getitem__(self, idx):
        prompt = item['prompt']
        
        # 运行时加载图片和 mask
        source_image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        
        # 应用变换
        image_tensor = self.image_transform(source_image)
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()  # 二值化
        
        # 创建遮罩图像
        masked_image = image_tensor * (1 - mask_tensor)
        
        return {
            'prompt': prompt,
            'image': image_tensor,
            'mask': mask_tensor,
            'masked_image': masked_image,
            'image_pil': source_image_resized,  # PIL 格式
            'mask_pil': mask_resized,
            'image_path': image_path,
            'mask_path': mask_path,
        }
```

---

## 4. 训练流程对比

### 4.1 Flux Fill 训练流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. 数据加载                                                              │
│    └── (prompt_embed, pooled_prompt_embeds, text_ids,                   │
│         masked_latents, mask_latents, caption)                          │
├─────────────────────────────────────────────────────────────────────────┤
│ 2. 采样阶段 (sample_reference_model)                                    │
│    ├── 初始化随机噪声 latents                                            │
│    ├── Pack masked_latents: [B, seq_len, 64]                            │
│    ├── 扩展 mask 维度并 pack: [B, 1, H, W] -> [B, seq_len, 256]          │
│    └── 迭代去噪 (run_sample_step_fill)                                   │
│        └── Transformer 输入 (特殊拼接方式):                               │
│            - Step 1: masked_image_latents = cat(masked_latents, mask)   │
│                      [B, seq_len, 64] + [B, seq_len, 256]               │
│                      = [B, seq_len, 320]                                │
│            - Step 2: hidden_states = cat(latents, masked_image_latents) │
│                      [B, seq_len, 64] + [B, seq_len, 320]               │
│                      = [B, seq_len, 384]                                │
├─────────────────────────────────────────────────────────────────────────┤
│ 3. 奖励计算                                                              │
│    └── reward = reward_model(decoded_image, caption)                    │
├─────────────────────────────────────────────────────────────────────────┤
│ 4. GRPO 训练                                                             │
│    └── 与文生图类似，但需要维护 masked_latents 和 mask_latents            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 SD1.5 Inpainting 训练流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. 数据加载 (运行时处理)                                                 │
│    └── (prompt, image_pil, mask_pil, image_path, mask_path, ...)        │
├─────────────────────────────────────────────────────────────────────────┤
│ 2. 采样阶段 (pipeline_inpainting_with_logprob)                          │
│    ├── 使用 pipeline 预处理 mask 和图像                                   │
│    ├── 运行时编码 prompt: prompt_embeds = text_encoder(tokenized)       │
│    ├── 准备 mask 和 masked_image_latents                                 │
│    └── 迭代去噪 (DDIM)                                                   │
│        └── UNet 输入:                                                    │
│            - latent_model_input = cat([latents, mask, masked_img], dim=1)│
│              [B, 4, H, W] + [B, 1, H, W] + [B, 4, H, W]                 │
│              = [B, 9, H, W]                                             │
│            - encoder_hidden_states: prompt_embeds                        │
├─────────────────────────────────────────────────────────────────────────┤
│ 3. 奖励计算                                                              │
│    └── reward = reward_model(decoded_image, prompt)                     │
├─────────────────────────────────────────────────────────────────────────┤
│ 4. GRPO 训练                                                             │
│    └── 策略梯度更新                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 采样过程对比

### 5.1 Flux Fill - run_sample_step_fill (拼接 mask)

```python
def run_sample_step_fill(args, input_latents, progress_bar, sigma_schedule,
                         transformer, encoder_hidden_states, pooled_prompt_embeds,
                         text_ids, image_ids, masked_latents, mask_latents,
                         grpo_sample):
    """Fill 采样 - 在特征维度拼接 mask"""
    latents = input_latents
    
    for i in progress_bar:
        # Step 1: masked_latents 和 mask 在特征维度拼接
        # masked_latents: [B, seq_len, 64]
        # mask_latents:   [B, seq_len, 256]
        masked_image_latents = torch.cat((masked_latents, mask_latents), dim=-1)
        # Result: [B, seq_len, 320]
        
        # Step 2: latents 和 masked_image_latents 在特征维度拼接
        # latents: [B, seq_len, 64]
        hidden_states = torch.cat((latents, masked_image_latents), dim=2)
        # Result: [B, seq_len, 384]
        
        pred = transformer(
            hidden_states=hidden_states,  # [B, seq_len, 384]
            encoder_hidden_states=encoder_hidden_states,
            img_ids=image_ids,  # 不需要拼接
            # ...
        )[0]
        
        # 输出直接对应 latents
        latents = latents + pred * dt
```

### 5.2 SD1.5 Inpainting - pipeline_inpainting_with_logprob

```python
def pipeline_inpainting_with_logprob(pipeline, prompt_embeds, negative_prompt_embeds,
                                     image, mask_image, ...):
    """SD1.5 Inpainting - 在通道维度拼接"""
    
    # 预处理 mask 和图像 (运行时)
    init_image = pipeline.image_processor.preprocess(image)
    mask_condition = pipeline.mask_processor.preprocess(mask_image)
    
    # 创建遮罩图像
    masked_image = init_image * (mask_condition < 0.5)
    
    # 准备 mask 和 masked_image latents
    mask, masked_image_latents = pipeline.prepare_mask_latents(
        mask_condition, masked_image, batch_size, height, width, ...
    )
    
    for i, t in enumerate(timesteps):
        # 扩展 latents 用于 CFG
        latent_model_input = torch.cat([latents] * 2)
        
        # 关键: 在通道维度拼接 mask 和 masked_image
        # latent:             [2B, 4, H, W]
        # mask:               [2B, 1, H, W]
        # masked_image_latents: [2B, 4, H, W]
        latent_model_input = torch.cat(
            [latent_model_input, mask, masked_image_latents], dim=1
        )
        # Result: [2B, 9, H, W]
        
        noise_pred = pipeline.unet(
            latent_model_input,  # 9 通道输入
            t,
            encoder_hidden_states=...,
        )[0]
```

---

## 6. Transformer 输入构建

### 6.1 核心差异总结

| 模型 | hidden_states 构建 | 维度 |
|------|-------------------|------|
| **Flux Fill** | `cat(latents, cat(masked_latents, mask, dim=-1), dim=2)` | `[B, seq_len, 384]` |
| **SD1.5 Inpainting** | `cat([latents, mask, masked_image], dim=1)` | `[B, 9, H, W]` |

### 6.2 Image IDs 处理

```python
# Flux Fill: 标准 image_ids
def prepare_latent_image_ids(height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = torch.arange(height)[:, None]  # y 坐标
    latent_image_ids[..., 2] = torch.arange(width)[None, :]   # x 坐标
    return latent_image_ids.reshape(height * width, 3)
```

---

## 7. 总结

1. **Flux Fill**: 需要 mask 和遮罩图像，通过 **特征维度拼接** 实现条件控制
2. **SD1.5 Inpainting**: 传统 UNet 架构，通过 **通道维度拼接** (9通道输入) 实现条件控制

这些差异反映了不同任务对条件信息注入方式的不同需求，以及 Transformer (Flux) 和 UNet (SD1.5) 架构的不同设计思路。
