#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

CKPT_DIR="../checkpoints"
KODAK="../kodak"
TECNICK="../tecnick_flat"
CITY="/home/common/EF-LIC/datasets/cityscapes/leftImg8bit/val/frankfurt"
MODEL_SIZE=64

echo "=== Step 1: Evaluation ==="
python evaluate_tcm.py \
    --datasets kodak tecnick cityscapes \
    --ckpt-dir "$CKPT_DIR" \
    --kodak-dir "$KODAK" \
    --tecnick-dir "$TECNICK" \
    --cityscapes-dir "$CITY" \
    --model-size "$MODEL_SIZE" \
    --max-cityscapes 100

echo
echo "=== Step 2: YOLOS comparison ==="
python yolos_batch_tcm.py \
    --datasets kodak tecnick cityscapes \
    --ckpt-dir "$CKPT_DIR" \
    --kodak-dir "$KODAK" \
    --tecnick-dir "$TECNICK" \
    --cityscapes-dir "$CITY" \
    --model-size "$MODEL_SIZE" \
    --num-images 10

echo
echo "Done."
