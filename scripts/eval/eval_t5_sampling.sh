#!/bin/bash
# Sampling (T=0.7, top-p=0.9, N=8) + multi-signal selection for a FLAN-T5 checkpoint.
# Set USE_PARSER=1 to enable the parser-conditioned selection variants.
#
# [Usage]  USE_PARSER=1  MODEL=runs/checkpoints/flan-t5-base-sft \
#            bash scripts/eval/eval_t5_sampling.sh

set -euo pipefail

MODEL="${MODEL:?Set MODEL.}"
TAG="${TAG:-$(basename "${MODEL}")_sample_n8}"
PARSER_ARG=""; [ "${USE_PARSER:-0}" = "1" ] && PARSER_ARG="--parser src/inference/vm_pseudocode_parser.mjs"

OUT_DIR="runs/eval_outputs"
mkdir -p "${OUT_DIR}"
PRED="${OUT_DIR}/${TAG}_predictions.jsonl"

python3 src/inference/generate_t5_predictions_select.py \
  --test data/test_sac_primary_subset_800.jsonl \
  --model "${MODEL}" \
  --out "${PRED}" \
  --decoding sampling \
  --num-candidates 8 --temperature 0.7 --top-p 0.9 --seed 42 \
  --max-source 256 --max-new-tokens 256 \
  ${PARSER_ARG}

eval_one() {
  local field="$1" name="$2"
  echo "=== ${name} ==="
  python3 src/inference/evaluate_predictions.py \
    --pred-file "${PRED}" --prediction-field "${field}" --postprocess \
    | tee "${OUT_DIR}/${TAG}_metrics_${name}.txt"
}

eval_one prediction            first
eval_one prediction_likelihood likelihood
eval_one prediction_reranked   sac
if [ "${USE_PARSER:-0}" = "1" ]; then
  eval_one prediction_parse_likelihood parse_likelihood
  eval_one prediction_parse_sac        parse_sac
fi
