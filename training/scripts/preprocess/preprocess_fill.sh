#!/bin/bash
# Preprocessing script for Flux Fill GRPO training
# Generates T5 embeddings and VAE latents

set -e

# Configuration
INPUT_JSONL="./data/fill_train.jsonl"
OUTPUT_DIR="./data/fill_preprocessed"
MODEL_PATH="black-forest-labs/FLUX.1-Fill-dev"  # Or local path
IMAGE_SIZE=512
MAX_LENGTH=512
DEVICE="cuda"

echo "=== Flux Fill Data Preprocessing ==="
echo "Input: ${INPUT_JSONL}"
echo "Output: ${OUTPUT_DIR}"
echo "Model: ${MODEL_PATH}"

python fastvideo/data_preprocess/preprocess_flux_kontext_fill_embeddings.py \
    --task fill \
    --input_jsonl ${INPUT_JSONL} \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${MODEL_PATH} \
    --image_size ${IMAGE_SIZE} \
    --max_length ${MAX_LENGTH} \
    --device ${DEVICE}

echo "=== Preprocessing Complete ==="
echo "Preprocessed data saved to: ${OUTPUT_DIR}"
echo "Use ${OUTPUT_DIR}/metadata.json for training"
