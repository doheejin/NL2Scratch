#!/bin/bash
# Greedy decoding for a FLAN-T5 (seq2seq) checkpoint on the SAC-primary 800 subset.
#
# [Usage]  MODEL=runs/checkpoints/flan-t5-base-sft  bash scripts/eval/eval_t5_greedy.sh

set -euo pipefail

MODEL="${MODEL:?Set MODEL (HF repo id or local checkpoint path).}"
TAG="${TAG:-$(basename "${MODEL}")_greedy}"

OUT_DIR="runs/eval_outputs"
mkdir -p "${OUT_DIR}"
PRED="${OUT_DIR}/${TAG}_predictions.jsonl"
METRICS="${OUT_DIR}/${TAG}_metrics.txt"

python3 src/inference/generate_t5_predictions_select.py \
  --test data/test_sac_primary_subset_800.jsonl \
  --model "${MODEL}" \
  --out "${PRED}" \
  --decoding greedy \
  --max-source 256 --max-new-tokens 256

python3 src/inference/evaluate_predictions.py \
  --pred-file "${PRED}" \
  --prediction-field prediction \
  --postprocess \
  | tee "${METRICS}"
