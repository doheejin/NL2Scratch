#!/bin/bash
# SFT of FLAN-T5 (encoder-decoder seq2seq).
# Source = NL, target = pseudocode joined by newlines.
# Loads the Heejindo/nl2scratch dataset directly from HuggingFace.
#
#   Usage:  bash scripts/sft/flan_sft.sh

set -euo pipefail

MODEL="${MODEL:-google/flan-t5-base}"
TAG="${TAG:-flan-t5-base-sft}"
DATA_DIR="runs/sft_data/${TAG}"
CKPT_DIR="runs/checkpoints/${TAG}"
mkdir -p "${DATA_DIR}" "${CKPT_DIR}"

[ -f "${DATA_DIR}/train.jsonl" ] || python3 src/training/prepare_sft_data.py --output-dir "${DATA_DIR}"

python3 src/training/train_t5.py \
  --train "${DATA_DIR}/train.jsonl" \
  --val   "${DATA_DIR}/val.jsonl" \
  --model "${MODEL}" \
  --out   "${CKPT_DIR}" \
  --bsz 4 --epochs 2 \
  --max-source 256 --max-target 256 \
  --lr 1e-4 --fp16
