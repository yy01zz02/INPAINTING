#!/bin/bash
# PrefPaint baseline inference script
# Run inpainting inference using PrefPaint model for benchmark evaluation

set -e

# ============================================================================
# Environment Variables (set these before running)
# ============================================================================
# export PREFPAINT_MODEL_PATH="/path/to/prefpaint/model"
# export RLBENCH_DIR="/path/to/rlbench/dataset"
# export OUTPUT_DIR="/path/to/output/prefpaint"

echo "Running PrefPaint baseline inference..."
echo "Model: ${PREFPAINT_MODEL_PATH}"
echo "Dataset: ${RLBENCH_DIR}"
echo "Output: ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} uv run python rlbench_prefpaint.py \
    --model_path "${PREFPAINT_MODEL_PATH}" \
    --base_dir "${RLBENCH_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 1234

echo "PrefPaint inference complete!"
