#!/bin/bash
# Greedy decoding for a causal-LM checkpoint on the SAC-primary 800 subset.
#
# [Usage]  Qwen:   MODEL=satechi/qwen2.5-7b-sft-final-lr5e5-nl2scratch \
#            BASE_MODEL=Qwen/Qwen2.5-7B-Instruct  HF_TOKEN=hf_xxx \
#            bash scripts/eval/eval_causal_greedy.sh
# [Usage]  Llama:  MODEL=runs/checkpoints/llama-3.1-8b-instruct-sft \
#            BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct  HF_TOKEN=hf_xxx \
#            bash scripts/eval/eval_causal_greedy.sh

set -euo pipefail

MODEL="${MODEL:?Set MODEL (HF repo id or local adapter/checkpoint path).}"
TAG="${TAG:-$(basename "${MODEL}")_greedy}"
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN
BASE_ARG=""; [ -n "${BASE_MODEL:-}" ] && BASE_ARG="--base-model ${BASE_MODEL}"

OUT_DIR="runs/eval_outputs"
mkdir -p "${OUT_DIR}"
PRED="${OUT_DIR}/${TAG}_predictions.jsonl"
METRICS="${OUT_DIR}/${TAG}_metrics.txt"

python3 src/inference/generate_predictions_select.py \
  --test data/test_sac_primary_subset_800.jsonl \
  --model "${MODEL}" ${BASE_ARG} \
  --out "${PRED}" \
  --decoding greedy \
  --max-new-tokens 384

python3 src/inference/evaluate_predictions.py \
  --pred-file "${PRED}" \
  --prediction-field prediction \
  --postprocess \
  | tee "${METRICS}"
