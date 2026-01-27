from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2
import uvicorn
from client_metrics import MetricsCalculator

app = FastAPI(title="Image Reward API")

# 初始化MetricsCalculator
device = 'cuda' if torch.cuda.is_available() else 'cpu'
metrics_calculator = MetricsCalculator(device=device)


class RewardRequest(BaseModel):
    """请求模型"""
    image_b_base64: str  # 生成的完整图片
    mask_base64: str     # mask图片
    prompt: str          # 文本提示


class RewardResponse(BaseModel):
    """响应模型"""
    boundary_score: float
    hps_score: float
    clip_score: float
    mask_ratio: float


def decode_base64_image(base64_str: str) -> Image.Image:
    """解码base64图片"""
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


def extract_mask_region(mask_img: np.ndarray, inpainted_img: np.ndarray, min_size: int = 224):
    """
    提取mask区域并按需要扩展到最小尺寸
    
    Args:
        mask_img: mask图片 (灰度图, numpy array)
        inpainted_img: 完整生成图片 (RGB, numpy array)
        min_size: 最小裁切尺寸
        
    Returns:
        cropped_image: 裁切后的图片区域 (PIL Image)
        cropped_mask: 裁切后的mask (PIL Image)
        crop_coords: 裁切坐标 (x, y, w, h)
    """
    # 获取原图尺寸
    img_height, img_width = inpainted_img.shape[:2]
    
    # 找到mask的边界框
    coords = cv2.findNonZero(mask_img)
    if coords is None:
        raise ValueError("Mask is empty")
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # 确保mask区域最小为min_size x min_size
    if w < min_size or h < min_size:
        # 计算需要扩展的尺寸
        target_w = max(min_size, w)
        target_h = max(min_size, h)
        
        expand_w = target_w - w
        expand_h = target_h - h
        
        # 向四周均匀扩展
        expand_left = expand_w // 2
        expand_right = expand_w - expand_left
        expand_top = expand_h // 2
        expand_bottom = expand_h - expand_top
        
        # 计算新的边界
        new_x = x - expand_left
        new_y = y - expand_top
        new_w = w + expand_left + expand_right
        new_h = h + expand_top + expand_bottom
        
        # 处理左边界溢出
        if new_x < 0:
            overflow = -new_x
            new_x = 0
            new_w = min(new_w + overflow, img_width)
        
        # 处理右边界溢出
        if new_x + new_w > img_width:
            overflow = (new_x + new_w) - img_width
            new_w = img_width - new_x
            new_x = max(0, new_x - overflow)
            new_w = min(target_w, img_width)
        
        # 处理上边界溢出
        if new_y < 0:
            overflow = -new_y
            new_y = 0
            new_h = min(new_h + overflow, img_height)
        
        # 处理下边界溢出
        if new_y + new_h > img_height:
            overflow = (new_y + new_h) - img_height
            new_h = img_height - new_y
            new_y = max(0, new_y - overflow)
            new_h = min(target_h, img_height)
        
        # 更新坐标
        x, y, w, h = new_x, new_y, new_w, new_h
    
    # 裁切图片
    cropped_image = inpainted_img[y:y+h, x:x+w]
    
    # 转换为PIL Image
    cropped_image_pil = Image.fromarray(cropped_image)
    
    return cropped_image_pil


def calculate_mask_ratio(mask_img: np.ndarray, image_shape: tuple) -> float:
    """
    计算mask占整个图片的比例
    
    Args:
        mask_img: mask图片 (灰度图)
        image_shape: 原图尺寸 (height, width)
        
    Returns:
        ratio: mask占比 (0-1之间)
    """
    return np.sum(mask_img > 127) / (image_shape[0] * image_shape[1])
    


@app.post("/calculate_reward", response_model=RewardResponse)
async def calculate_reward(request: RewardRequest):
    try:
        # 1. 解码图片
        image_b = decode_base64_image(request.image_b_base64)
        mask = decode_base64_image(request.mask_base64)
        
        # 2. 转换为numpy数组
        image_b_np = np.array(image_b.convert('RGB'))
        mask_np = np.array(mask.convert('L'))
        
        # 3. 计算mask占比 (a值)
        mask_ratio = calculate_mask_ratio(mask_np, image_b_np.shape[:2])
        
        # 4. 裁切mask区域得到图片c
        try:
            image_c = extract_mask_region(
                mask_np, 
                image_b_np
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # 5. 计算各项分数
        # 边界平滑度分数 (使用完整图片b和mask)
        boundary_score = metrics_calculator.calculate_boundary_smoothness(
            image_b, 
            mask
        )
        
        # HPS分数 (使用裁切后的图片c)
        hps_score = metrics_calculator.calculate_hpsv21_score(
            image_c, 
            request.prompt
        )
        
        # CLIP分数 (使用裁切后的图片c)
        clip_score = metrics_calculator.calculate_clip_similarity(
            image_c, 
            request.prompt
        )
        
        return RewardResponse(
            boundary_score=float(boundary_score),
            hps_score=float(hps_score),
            clip_score=float(clip_score),
            mask_ratio=float(mask_ratio)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating reward: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "device": device,
        "cuda_available": torch.cuda.is_available()
    }


if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8169,
        log_level="info"
    )