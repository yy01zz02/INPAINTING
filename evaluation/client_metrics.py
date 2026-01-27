"""
Metrics Calculator Client
Provides unified interface for computing various image quality metrics.
Connects to reward model servers via HTTP API and computes local metrics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image
import math
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Server URLs for reward models
URLS = {
    "aesthetic": "http://localhost:8161/score",
    "clip": "http://localhost:8162/score",
    "hpsv2": "http://localhost:8163/score",
    "hpsv3": "http://localhost:8164/score",
    "imagereward": "http://localhost:8165/score",
    "pickscore": "http://localhost:8166/score",
}


class BoundaryReward(nn.Module):
    """Computes boundary smoothness using gradient discontinuity."""
    
    def __init__(self, device='cuda', kernel_size=5):
        super().__init__()
        self.device = device
        self.kernel_size = kernel_size
        
        self.sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device
        ).float().view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device
        ).float().view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    def get_boundary_masks(self, mask):
        k = self.kernel_size
        pad = k // 2
        dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
        outer_boundary = dilated - mask
        eroded = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=pad)
        inner_boundary = mask - eroded
        return inner_boundary, outer_boundary

    def compute_gradient_discontinuity(self, images, masks):
        grad_x = F.conv2d(images, self.sobel_x, padding=1, groups=3)
        grad_y = F.conv2d(images, self.sobel_y, padding=1, groups=3)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8).mean(dim=1, keepdim=True)
        
        inner_boundary, outer_boundary = self.get_boundary_masks(masks)
        inner_grad = (grad_mag * inner_boundary).sum(dim=[2,3]) / (inner_boundary.sum(dim=[2,3]) + 1e-6)
        outer_grad = (grad_mag * outer_boundary).sum(dim=[2,3]) / (outer_boundary.sum(dim=[2,3]) + 1e-6)
        grad_diff = torch.abs(inner_grad - outer_grad).squeeze()
        return -grad_diff

    def forward(self, images, masks):
        return self.compute_gradient_discontinuity(images, masks)


class MetricsCalculator:
    """Unified metrics calculator for inpainting evaluation."""
    
    def __init__(self, device, server_urls=URLS):
        self.device = device
        self.server_urls = server_urls
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        self.boundary_reward = BoundaryReward(device=device)

    def _encode_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _call_service(self, metric_name, payload):
        url = self.server_urls.get(metric_name)
        if not url:
            return 0.0
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()['score']
        except Exception as e:
            print(f"Error calling {metric_name}: {e}")
            return 0.0

    # Remote metrics (via server API)
    def calculate_image_reward(self, image, prompt):
        return self._call_service("imagereward", {"image_base64": self._encode_image(image), "prompt": prompt})

    def calculate_hpsv21_score(self, image, prompt):
        return self._call_service("hpsv2", {"image_base64": self._encode_image(image), "prompt": prompt})

    def calculate_aesthetic_score(self, img):
        return self._call_service("aesthetic", {"image_base64": self._encode_image(img)})

    def calculate_clip_similarity(self, img, txt):
        return self._call_service("clip", {"image_base64": self._encode_image(img), "prompt": txt})
    
    def calculate_pick_score(self, img, txt):
        return self._call_service("pickscore", {"image_base64": self._encode_image(img), "prompt": txt})

    def calculate_hpsv3_score(self, img, txt):
        return self._call_service("hpsv3", {"image_base64": self._encode_image(img), "prompt": txt})

    # Local metrics
    def calculate_psnr(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255.
        img_gt = np.array(img_gt).astype(np.float32) / 255.
        
        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask
            size = mask.sum()
        else:
            size = img_pred.size
        
        mse = ((img_pred - img_gt) ** 2).sum() / size
        if mse < 1e-10:
            return 100
        return 20 * math.log10(1.0 / math.sqrt(mse))
    
    def calculate_lpips(self, img_gt, img_pred, mask=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255.
        img_gt = np.array(img_gt).astype(np.float32) / 255.

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask
            
        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        score = self.lpips_metric(img_pred_tensor * 2 - 1, img_gt_tensor * 2 - 1)
        return score.cpu().item()
    
    def calculate_mse(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255.
        img_gt = np.array(img_gt).astype(np.float32) / 255.

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask
            size = mask.sum()
        else:
            size = img_pred.size
        
        mse = ((img_pred - img_gt) ** 2).sum() / size
        return mse.item() if hasattr(mse, 'item') else float(mse)

    def calculate_boundary_smoothness(self, image, mask):
        """Compute boundary smoothness score (higher = smoother)."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.NEAREST)
        
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
        
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        mask_tensor = mask_tensor.unsqueeze(0).to(self.device)
        mask_tensor = (mask_tensor > 0.5).float()
        
        with torch.no_grad():
            score = self.boundary_reward(img_tensor, mask_tensor)
        
        return score.item()
