#!/bin/bash
# SFT of Llama-3.1-8B-Instruct using bf16 LoRA on a single ~40 GB GPU.
# Loads the Heejindo/nl2scratch dataset directly from HuggingFace.
#
# [Usage]  HF_TOKEN=hf_xxx  bash scripts/sft/llama_sft.sh

set -euo pipefail

: "${HF_TOKEN:?Export HF_TOKEN for gated base models.}"
export HF_TOKEN

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TAG="${TAG:-llama-3.1-8b-instruct-sft}"
DATA_DIR="runs/sft_data/${TAG}"
CKPT_DIR="runs/checkpoints/${TAG}"
mkdir -p "${DATA_DIR}" "${CKPT_DIR}"

[ -f "${DATA_DIR}/train.jsonl" ] || python3 src/training/prepare_sft_data.py --output-dir "${DATA_DIR}"

RESUME_ARGS=(); shopt -s nullglob
mapfile -t CKPTS < <(printf '%s\n' "${CKPT_DIR}"/checkpoint-* | sort -t- -k2 -n)
shopt -u nullglob
if [ ${#CKPTS[@]} -gt 0 ]; then
  LAST="${CKPTS[-1]}"
  echo "Resuming from ${LAST}"
  RESUME_ARGS=(--resume-from-checkpoint "${LAST}")
fi

python3 src/training/train_causal_sft.py \
  --train "${DATA_DIR}/train.jsonl" \
  --val   "${DATA_DIR}/val.jsonl" \
  --model "${MODEL}" \
  --out   "${CKPT_DIR}" \
  --quantization none \
  --bsz 8 --grad-accum 2 --lr 5e-5 --epochs 2 \
  --max-seq-length 768 --dtype bf16 \
  --save-steps 1000 --eval-steps 1000 --logging-steps 20 \
  "${RESUME_ARGS[@]}"
