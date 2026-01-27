import os
import json
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from client_metrics import MetricsCalculator
import torch
from pathlib import Path

def find_image(base_path, name):
    """Find image with either .jpg or .png extension"""
    for ext in ['.jpg', '.png']:
        path = base_path / f"{name}{ext}"
        if path.exists():
            return path
    return None

def load_mask(mask_path):
    """Load mask and convert to binary (0 or 1)"""
    mask = np.array(Image.open(mask_path).convert('L'))
    mask = (mask > 127).astype(np.uint8)
    return 1 - mask[:, :, np.newaxis]  # Invert mask

parser = argparse.ArgumentParser()
parser.add_argument('--bench_dir', type=str, required=True, help="Path to brushbench or editbench directory")
parser.add_argument('--model_name', type=str, required=True, help="Model name (subfolder in results)")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Evaluation running on {device}")

bench_dir = Path(args.bench_dir)
bench_name = bench_dir.name  # brushbench or editbench
model_name = args.model_name

# Paths
originals_dir = bench_dir / "originals"
masks_dir = bench_dir / "masks"
prompts_dir = bench_dir / "prompts"
results_dir = bench_dir / "results" / model_name
output_dir = bench_dir / "evaluations"
output_dir.mkdir(exist_ok=True)

# Initialize metrics calculator
metrics_calculator = MetricsCalculator(device)

# Get all image names (without extension)
image_names = set([p.stem for p in originals_dir.glob('*') if p.suffix in ['.jpg', '.png']])

evaluation_df = pd.DataFrame(columns=[
    'Image ID',
    'Image Reward', 'HPS V2.1', 'Aesthetic Score', 'CLIP Score', 'PickScore', 'HPS v3',
    'PSNR', 'LPIPS', 'MSE'
])

print(f"Starting evaluation for {bench_name} - {model_name}")
print(f"Found {len(image_names)} images")

for name in sorted(image_names):
    # Find all required files
    original_path = find_image(originals_dir, name)
    mask_path = find_image(masks_dir, name)
    result_path = find_image(results_dir, name)
    prompt_path = prompts_dir / f"{name}.txt"
    
    # Check if all files exist
    if not all([original_path, mask_path, result_path, prompt_path]):
        print(f"Warning: Missing files for {name}. Skipping.")
        continue
    
    print(f"Evaluating {name}...")
    
    # Load images and prompt
    src_image = Image.open(original_path).convert("RGB")
    tgt_image = Image.open(result_path).convert("RGB")
    size = src_image.size[0]  # Assume square image
    
    # # Ensure same size
    # tgt_image = tgt_image.resize((size, size))
    
    # Load mask
    mask = load_mask(mask_path)
    
    # Load prompt
    with open(prompt_path, 'r') as f:
        prompt = f.read().strip()
    
    evaluation_result = [name]
    
    # Calculate Metrics
    evaluation_result.append(metrics_calculator.calculate_image_reward(tgt_image, prompt))
    evaluation_result.append(metrics_calculator.calculate_hpsv21_score(tgt_image, prompt))
    evaluation_result.append(metrics_calculator.calculate_aesthetic_score(tgt_image))
    evaluation_result.append(metrics_calculator.calculate_clip_similarity(tgt_image, prompt))
    evaluation_result.append(metrics_calculator.calculate_pick_score(tgt_image, prompt))
    evaluation_result.append(metrics_calculator.calculate_hpsv3_score(tgt_image, prompt))
    evaluation_result.append(metrics_calculator.calculate_psnr(src_image, tgt_image, mask))
    evaluation_result.append(metrics_calculator.calculate_lpips(src_image, tgt_image, mask))
    evaluation_result.append(metrics_calculator.calculate_mse(src_image, tgt_image, mask))
    
    evaluation_df.loc[len(evaluation_df.index)] = evaluation_result

# Save Results
print("\nAveraged evaluation results:")
averaged_results = evaluation_df.mean(numeric_only=True)
print(averaged_results)

output_prefix = f"{bench_name}_{model_name}"
averaged_results.to_csv(output_dir / f"{output_prefix}_summary.csv")
evaluation_df.to_csv(output_dir / f"{output_prefix}_detailed.csv")

print(f"\nResults saved in {output_dir}")
print(f"- {output_prefix}_summary.csv")
print(f"- {output_prefix}_detailed.csv")
