"""
Boundary Smoothness Calculator
Computes gradient discontinuity at mask boundaries using Sobel operators.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class BoundaryReward(nn.Module):
    """Computes boundary smoothness score based on gradient discontinuity."""
    
    def __init__(self, device='cuda', kernel_size=5):
        super().__init__()
        self.device = device
        self.kernel_size = kernel_size
        
        # Sobel operators for gradient computation
        self.sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
            device=device
        ).float().view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
            device=device
        ).float().view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    def get_boundary_masks(self, mask):
        """Separate inner and outer boundary regions."""
        k = self.kernel_size
        pad = k // 2
        
        # Outer boundary (outside mask)
        dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
        outer_boundary = dilated - mask
        
        # Inner boundary (inside mask)
        eroded = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=pad)
        inner_boundary = mask - eroded
        
        return inner_boundary, outer_boundary

    def compute_gradient_discontinuity(self, images, masks):
        """Compute gradient difference between inner and outer boundaries."""
        # Compute gradients
        grad_x = F.conv2d(images, self.sobel_x, padding=1, groups=3)
        grad_y = F.conv2d(images, self.sobel_y, padding=1, groups=3)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8).mean(dim=1, keepdim=True)
        
        # Get boundary regions
        inner_boundary, outer_boundary = self.get_boundary_masks(masks)
        
        # Compute average gradients at boundaries
        inner_grad = (grad_mag * inner_boundary).sum(dim=[2,3]) / (inner_boundary.sum(dim=[2,3]) + 1e-6)
        outer_grad = (grad_mag * outer_boundary).sum(dim=[2,3]) / (outer_boundary.sum(dim=[2,3]) + 1e-6)
        
        # Gradient difference (smaller = smoother boundary)
        grad_diff = torch.abs(inner_grad - outer_grad).squeeze()
        
        # Convert to reward (smaller difference = higher score)
        score = -grad_diff
        return score

    def forward(self, images, masks):
        return self.compute_gradient_discontinuity(images, masks)


def calculate_boundary_score(image_path, mask_path, device='cuda'):
    """Calculate boundary smoothness score for a single image."""
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    img_pil = Image.open(image_path).convert('RGB')
    mask_pil = Image.open(mask_path).convert('L')
    
    # Resize mask to match image
    if img_pil.size != mask_pil.size:
        mask_pil = mask_pil.resize(img_pil.size, Image.NEAREST)
    
    # Convert to tensors
    transform = transforms.ToTensor()
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    mask_tensor = transform(mask_pil).unsqueeze(0).to(device)
    mask_tensor = (mask_tensor > 0.5).float()
    
    # Compute score
    reward_fn = BoundaryReward(device=device)
    score = reward_fn(img_tensor, mask_tensor)
    
    return score.item()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--mask', type=str, required=True)
    args = parser.parse_args()
    
    score = calculate_boundary_score(args.image, args.mask)
    print(f"Boundary smoothness score: {score:.4f}")
