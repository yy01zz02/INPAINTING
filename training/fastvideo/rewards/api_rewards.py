# Unified reward functions for GRPO training.
# Reward functions are encapsulated here for easy switching between different reward models.
# To change reward model implementation, only modify this file.

import torch
import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image
from typing import Union, List, Optional


# ============================================================================
# Configuration - Modify these URLs if your reward API endpoints change
# ============================================================================
REWARD_API_CONFIG = {
    "clip": {
        "url": "http://localhost:8162/score",
        "timeout": 30,
    },
    "hpsv2": {
        "url": "http://localhost:8163/score",
        "timeout": 30,
    },
    "inpainting": {
        "url": "http://127.0.0.1:8169/calculate_reward",
        "timeout": 60,
    },
}


# ============================================================================
# Base Reward Class
# ============================================================================
class BaseReward:
    """Base class for reward functions."""
    
    def __init__(self, device):
        self.device = device
    
    def _encode_image(self, image: Union[Image.Image, np.ndarray, str]) -> str:
        """Encode image to base64 string."""
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _call_api(self, api_name: str, payload: dict) -> float:
        """Call reward API service."""
        config = REWARD_API_CONFIG.get(api_name)
        if not config:
            print(f"API config for {api_name} not found.")
            return 0.0
        
        try:
            resp = requests.post(
                config["url"], 
                json=payload, 
                timeout=config["timeout"]
            )
            resp.raise_for_status()
            return resp.json()['score']
        except requests.exceptions.Timeout:
            print(f"Timeout calling {api_name} API")
            return 0.0
        except requests.exceptions.RequestException as e:
            print(f"Error calling {api_name} API: {e}")
            return 0.0
        except Exception as e:
            print(f"Error calculating {api_name} score: {e}")
            return 0.0
    
    def compute_reward(
        self, 
        image: Union[Image.Image, np.ndarray, str], 
        prompt: str
    ) -> float:
        """Compute reward for a single image-prompt pair. Override in subclass."""
        raise NotImplementedError
    
    def __call__(
        self, 
        images: List[Union[Image.Image, np.ndarray, str]], 
        prompts: List[str]
    ) -> torch.Tensor:
        """Compute rewards for a batch of image-prompt pairs."""
        rewards = []
        for image, prompt in zip(images, prompts):
            score = self.compute_reward(image, prompt)
            rewards.append(score)
        return torch.tensor(rewards, device=self.device, dtype=torch.float32)


# ============================================================================
# CLIP-only Reward Function
# ============================================================================
class CLIPAPIReward(BaseReward):
    """CLIP-based reward function.
    
    Computes CLIP similarity score between generated image and text prompt.
    Uses API service for inference.
    """
    
    def __init__(self, device):
        super().__init__(device)
    
    def compute_reward(
        self, 
        image: Union[Image.Image, np.ndarray, str], 
        prompt: str
    ) -> float:
        """Compute CLIP similarity score."""
        payload = {
            "image_base64": self._encode_image(image),
            "prompt": prompt
        }
        return self._call_api("clip", payload)
    
    # Alias for backward compatibility
    def calculate_clip_similarity(self, image, prompt):
        return self.compute_reward(image, prompt)


# ============================================================================
# HPS-only Reward Function
# ============================================================================
class HPSAPIReward(BaseReward):
    """HPSv2-based reward function.
    
    Computes Human Preference Score v2 for generated image.
    Uses API service for inference.
    """
    
    def __init__(self, device):
        super().__init__(device)
    
    def compute_reward(
        self, 
        image: Union[Image.Image, np.ndarray, str], 
        prompt: str
    ) -> float:
        """Compute HPSv2 score."""
        payload = {
            "image_base64": self._encode_image(image),
            "prompt": prompt
        }
        return self._call_api("hpsv2", payload)
    
    # Alias for backward compatibility
    def calculate_hpsv2_score(self, image, prompt):
        return self.compute_reward(image, prompt)


# ============================================================================
# CLIP + HPS Combined Reward Function (1:1 ratio)
# ============================================================================
class CLIPHPSAPIReward(BaseReward):
    """Combined CLIP and HPSv2 reward function.
    
    Computes weighted combination of CLIP similarity and HPSv2 scores.
    Default ratio is 1:1 (clip_weight=0.5, hps_weight=0.5).
    """
    
    def __init__(self, device, clip_weight: float = 0.5, hps_weight: float = 0.5):
        super().__init__(device)
        
        # Normalize weights
        total = clip_weight + hps_weight
        self.clip_weight = clip_weight / total
        self.hps_weight = hps_weight / total
    
    def _get_clip_score(self, image, prompt) -> float:
        """Get CLIP similarity score."""
        payload = {
            "image_base64": self._encode_image(image),
            "prompt": prompt
        }
        return self._call_api("clip", payload)
    
    def _get_hps_score(self, image, prompt) -> float:
        """Get HPSv2 score."""
        payload = {
            "image_base64": self._encode_image(image),
            "prompt": prompt
        }
        return self._call_api("hpsv2", payload)
    
    def compute_reward(
        self, 
        image: Union[Image.Image, np.ndarray, str], 
        prompt: str
    ) -> float:
        """Compute combined CLIP + HPS reward."""
        clip_score = self._get_clip_score(image, prompt)
        hps_score = self._get_hps_score(image, prompt)
        return self.clip_weight * clip_score + self.hps_weight * hps_score
    
    # Aliases for backward compatibility
    def calculate_clip_similarity(self, image, prompt):
        return self._get_clip_score(image, prompt)
    
    def calculate_hpsv2_score(self, image, prompt):
        return self._get_hps_score(image, prompt)
    
    def calculate_combined_score(self, image, prompt):
        return self.compute_reward(image, prompt)


# ============================================================================
# Inpainting Reward Function with Group Normalization
# ============================================================================
class InpaintingAPIReward(BaseReward):
    """Inpainting reward function with boundary smoothness, HPS, and CLIP scores.
    
    This reward function calls an API that returns:
    - boundary_score: Boundary smoothness score
    - hps_score: HPSv2 score
    - clip_score: CLIP similarity score
    - mask_ratio: Ratio of mask area to total image area
    
    For GRPO training, use compute_reward_raw() to get raw scores, then call
    compute_group_normalized_rewards() to normalize across a group of generations.
    
    Final reward formula:
    reward = mask_ratio * (hps_score_norm + clip_score_norm) + (1 - mask_ratio) * boundary_score_norm
    """
    
    def __init__(self, device):
        super().__init__(device)
    
    def _encode_mask(self, mask: Union[Image.Image, np.ndarray, str]) -> str:
        """Encode mask image to base64 string."""
        if isinstance(mask, str):
            mask = Image.open(mask)
        elif isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        
        # Keep mask as grayscale
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        buffered = BytesIO()
        mask.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def compute_reward_raw(
        self, 
        image: Union[Image.Image, np.ndarray, str], 
        mask: Union[Image.Image, np.ndarray, str],
        prompt: str
    ) -> dict:
        """Compute raw reward scores from API.
        
        Args:
            image: Generated inpainted image
            mask: Mask image (white = inpainted region)
            prompt: Text prompt
            
        Returns:
            dict with keys: boundary_score, hps_score, clip_score, mask_ratio
        """
        config = REWARD_API_CONFIG.get("inpainting")
        if not config:
            print("Inpainting API config not found.")
            return {
                "boundary_score": 0.0,
                "hps_score": 0.0,
                "clip_score": 0.0,
                "mask_ratio": 0.5,
            }
        
        payload = {
            "image_b_base64": self._encode_image(image),
            "mask_base64": self._encode_mask(mask),
            "prompt": prompt
        }
        
        try:
            resp = requests.post(
                config["url"], 
                json=payload, 
                timeout=config["timeout"]
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "boundary_score": float(data.get("boundary_score", 0.0)),
                "hps_score": float(data.get("hps_score", 0.0)),
                "clip_score": float(data.get("clip_score", 0.0)),
                "mask_ratio": float(data.get("mask_ratio", 0.5)),
            }
        except requests.exceptions.Timeout:
            print("Timeout calling inpainting API")
            return {
                "boundary_score": 0.0,
                "hps_score": 0.0,
                "clip_score": 0.0,
                "mask_ratio": 0.5,
            }
        except requests.exceptions.RequestException as e:
            print(f"Error calling inpainting API: {e}")
            return {
                "boundary_score": 0.0,
                "hps_score": 0.0,
                "clip_score": 0.0,
                "mask_ratio": 0.5,
            }
        except Exception as e:
            print(f"Error calculating inpainting score: {e}")
            return {
                "boundary_score": 0.0,
                "hps_score": 0.0,
                "clip_score": 0.0,
                "mask_ratio": 0.5,
            }
    
    def compute_group_normalized_rewards(
        self,
        raw_scores_list: List[dict],
        num_generations: int
    ) -> torch.Tensor:
        """Compute group-normalized rewards for GRPO training.
        
        For each group of num_generations samples (from the same prompt),
        normalize boundary_score, hps_score, and clip_score within the group,
        then compute final reward as:
        reward = mask_ratio * (hps_score_norm + clip_score_norm) + (1 - mask_ratio) * boundary_score_norm
        
        Args:
            raw_scores_list: List of dicts from compute_reward_raw()
            num_generations: Number of generations per prompt
            
        Returns:
            Tensor of normalized rewards
        """
        n_samples = len(raw_scores_list)
        n_groups = n_samples // num_generations
        
        # Extract scores
        boundary_scores = torch.tensor([s["boundary_score"] for s in raw_scores_list], dtype=torch.float32)
        hps_scores = torch.tensor([s["hps_score"] for s in raw_scores_list], dtype=torch.float32)
        clip_scores = torch.tensor([s["clip_score"] for s in raw_scores_list], dtype=torch.float32)
        mask_ratios = torch.tensor([s["mask_ratio"] for s in raw_scores_list], dtype=torch.float32)
        
        # Group normalize each score type
        boundary_norm = torch.zeros_like(boundary_scores)
        hps_norm = torch.zeros_like(hps_scores)
        clip_norm = torch.zeros_like(clip_scores)
        
        for i in range(n_groups):
            start_idx = i * num_generations
            end_idx = (i + 1) * num_generations
            
            # Normalize boundary_score within group
            group_boundary = boundary_scores[start_idx:end_idx]
            boundary_mean = group_boundary.mean()
            boundary_std = group_boundary.std() + 1e-8
            boundary_norm[start_idx:end_idx] = (group_boundary - boundary_mean) / boundary_std
            
            # Normalize hps_score within group
            group_hps = hps_scores[start_idx:end_idx]
            hps_mean = group_hps.mean()
            hps_std = group_hps.std() + 1e-8
            hps_norm[start_idx:end_idx] = (group_hps - hps_mean) / hps_std
            
            # Normalize clip_score within group
            group_clip = clip_scores[start_idx:end_idx]
            clip_mean = group_clip.mean()
            clip_std = group_clip.std() + 1e-8
            clip_norm[start_idx:end_idx] = (group_clip - clip_mean) / clip_std
        
        # Compute final reward:
        # reward = mask_ratio * (hps_norm + clip_norm) + (1 - mask_ratio) * boundary_norm
        rewards = mask_ratios * (hps_norm + clip_norm) + (1 - mask_ratios) * boundary_norm
        
        return rewards.to(self.device)
    
    def compute_reward(
        self, 
        image: Union[Image.Image, np.ndarray, str], 
        prompt: str
    ) -> float:
        """Compute reward for a single image (without mask - uses dummy mask).
        
        Note: For proper inpainting reward, use compute_reward_raw() with mask,
        then compute_group_normalized_rewards() for GRPO training.
        """
        # Without mask, just return combined HPS + CLIP score
        # This is a fallback for compatibility
        return 0.0
    
    def compute_reward_with_mask(
        self, 
        image: Union[Image.Image, np.ndarray, str], 
        mask: Union[Image.Image, np.ndarray, str],
        prompt: str
    ) -> float:
        """Compute single reward with mask (no group normalization).
        
        Uses simple weighted combination without group normalization.
        For GRPO training, use compute_reward_raw() + compute_group_normalized_rewards().
        """
        raw = self.compute_reward_raw(image, mask, prompt)
        # Simple weighted combination for single-sample case
        mask_ratio = raw["mask_ratio"]
        return mask_ratio * (raw["hps_score"] + raw["clip_score"]) + (1 - mask_ratio) * raw["boundary_score"]


# ============================================================================
# Factory Function - Use this to get reward model by name
# ============================================================================
def get_reward_fn(reward_type: str, device):
    """Get reward function by type name.
    
    Args:
        reward_type: One of 'clip', 'clip_hps', 'inpainting'
        device: Torch device
        
    Returns:
        Reward function instance
        
    Example:
        reward_fn = get_reward_fn('clip', device)
        score = reward_fn.compute_reward(image, prompt)
        
        # Or for batch:
        scores = reward_fn(images, prompts)
        
        # For inpainting:
        reward_fn = get_reward_fn('inpainting', device)
        raw_scores = reward_fn.compute_reward_raw(image, mask, prompt)
        rewards = reward_fn.compute_group_normalized_rewards(raw_scores_list, num_generations)
    """
    reward_types = {
        'clip': CLIPAPIReward,
        'clip_api': CLIPAPIReward,
        'hps': HPSAPIReward,
        'hps_api': HPSAPIReward,
        'hpsv2': HPSAPIReward,
        'clip_hps': CLIPHPSAPIReward,
        'clip_hps_api': CLIPHPSAPIReward,
        'inpainting': InpaintingAPIReward,
        'inpainting_api': InpaintingAPIReward,
    }
    
    if reward_type not in reward_types:
        raise ValueError(
            f"Unknown reward type: {reward_type}. "
            f"Available types: {list(reward_types.keys())}"
        )
    
    return reward_types[reward_type](device)


# Backward compatibility alias
create_reward_model = get_reward_fn
