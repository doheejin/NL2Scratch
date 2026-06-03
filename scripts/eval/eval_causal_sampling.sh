#!/bin/bash
# Sampling (T=0.7, top-p=0.9, N=8) + multi-signal selection for a causal-LM
# checkpoint on the SAC-primary 800 subset. Emits one metric file per rule:
#   first / likelihood / SAC, plus parse→{likelihood,SAC} when USE_PARSER=1
# (Node.js required for the scratchblocks parser).
#
#   Qwen:   USE_PARSER=1  MODEL=satechi/qwen2.5-7b-sft-final-lr5e5-nl2scratch \
#           BASE_MODEL=Qwen/Qwen2.5-7B-Instruct  HF_TOKEN=hf_xxx \
#             bash scripts/eval/eval_causal_sampling.sh

set -euo pipefail

MODEL="${MODEL:?Set MODEL.}"
TAG="${TAG:-$(basename "${MODEL}")_sample_n8}"
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN
BASE_ARG=""; [ -n "${BASE_MODEL:-}" ] && BASE_ARG="--base-model ${BASE_MODEL}"
PARSER_ARG=""; [ "${USE_PARSER:-0}" = "1" ] && PARSER_ARG="--parser src/inference/vm_pseudocode_parser.mjs"

OUT_DIR="runs/eval_outputs"
mkdir -p "${OUT_DIR}"
PRED="${OUT_DIR}/${TAG}_predictions.jsonl"

python3 src/inference/generate_predictions_select.py \
  --test data/test_sac_primary_subset_800.jsonl \
  --model "${MODEL}" ${BASE_ARG} \
  --out "${PRED}" \
  --decoding sampling \
  --num-candidates 8 --temperature 0.7 --top-p 0.9 --seed 42 \
  --max-new-tokens 384 \
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
