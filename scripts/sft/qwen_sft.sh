#!/bin/bash
# SFT of Qwen2.5-7B-Instruct using 4-bit QLoRA on a single ~40 GB GPU.
# Loads the Heejindo/nl2scratch dataset directly from HuggingFace.
#
#   Usage:  HF_TOKEN=hf_xxx  bash scripts/sft/qwen_sft.sh

set -euo pipefail

: "${HF_TOKEN:?Export HF_TOKEN for gated base models.}"
export HF_TOKEN

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TAG="${TAG:-qwen2.5-7b-sft}"
DATA_DIR="runs/sft_data/${TAG}"
CKPT_DIR="runs/checkpoints/${TAG}"
mkdir -p "${DATA_DIR}" "${CKPT_DIR}"

# 1) Tokenize prompts (skipped if already done).
[ -f "${DATA_DIR}/train.jsonl" ] || python3 src/training/prepare_sft_data.py --output-dir "${DATA_DIR}"

# 2) Auto-resume from the latest checkpoint if any.
RESUME_ARGS=(); shopt -s nullglob
mapfile -t CKPTS < <(printf '%s\n' "${CKPT_DIR}"/checkpoint-* | sort -t- -k2 -n)
shopt -u nullglob
if [ ${#CKPTS[@]} -gt 0 ]; then
  LAST="${CKPTS[-1]}"
  echo "Resuming from ${LAST}"
  RESUME_ARGS=(--resume-from-checkpoint "${LAST}")
fi

# 3) SFT.
python3 src/training/train_causal_sft.py \
  --train "${DATA_DIR}/train.jsonl" \
  --val   "${DATA_DIR}/val.jsonl" \
  --model "${MODEL}" \
  --out   "${CKPT_DIR}" \
  --quantization 4bit \
  --bsz 8 --grad-accum 2 --lr 5e-5 --epochs 2 \
  --max-seq-length 768 --dtype bf16 \
  --save-steps 1000 --eval-steps 1000 --logging-steps 20 \
  "${RESUME_ARGS[@]}"
