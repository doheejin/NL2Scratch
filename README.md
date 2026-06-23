# NL2Scratch

Official code repository for the paper *"NL2Scratch: An Executable Benchmark and Evaluation for Block-Based Programming."*

This repository contains the **SFT**, **evaluation**, and **inference-time candidate selection (incl. SAC reranking)** code used in the paper. Dataset construction code is released separately.

## Repository layout

```
NL2Scratch/
├── data/
│   └── test_sac_primary_subset_800.jsonl     # SAC-primary diagnostic test set (800 ex.)
├── src/
│   ├── sac/
│   │   └── semantic_alignment_check.py        # Rule-based SAC scorer (NL ↔ pseudocode)
│   ├── training/
│   │   ├── prepare_sft_data.py                # NL+pseudocode → SFT JSONL
│   │   ├── train_causal_sft.py                # SFT for causal LMs (Qwen / Llama)
│   │   └── train_t5.py                        # SFT for encoder-decoder (FLAN-T5)
│   ├── inference/
│   │   ├── generate_predictions_select.py     # Causal-LM generation + multi-signal selection
│   │   ├── generate_t5_predictions_select.py  # FLAN-T5 variant
│   │   ├── evaluate_predictions.py            # EM / line-EM / token-F1 / SAC_nl / SAC_pseudo / parse rate
│   │   └── vm_pseudocode_parser.mjs           # Scratchblocks parser (Node)
│   └── prompting_baseline/
│       ├── run_icl.py                         # OpenAI in-context prompting baseline
│       └── results/                           # Saved prompting-baseline outputs
└── scripts/
    ├── sft/                                   # Bash launchers for SFT
    └── eval/                                  # Bash launchers for evaluation
```

## Setup

```bash
git clone <repo>; cd NL2Scratch
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Node 18+ for the scratchblocks parser (required by the parser-based selection rules).
node --version
```

All shell launchers under `scripts/` assume the working directory is the repository root (i.e. run them as `bash scripts/.../foo.sh` from `NL2Scratch/`).

## Data

The **SAC-primary 800 test subset** used for all main-paper evaluations is bundled
in `data/test_sac_primary_subset_800.jsonl` so the evaluation pipeline runs out of
the box.

The full **train / validation splits** are released as a HuggingFace dataset at
[`Heejindo/nl2scratch`](https://huggingface.co/datasets/Heejindo/nl2scratch).
The training launchers stream this dataset directly via
`src/training/prepare_sft_data.py`, so no manual download is required.

Each row uses the schema
```json
{"key": "<id>", "nl": "<natural-language instruction>", "pseudocode": ["block1", "    block2", ...]}
```
where `pseudocode` is a list of normalized scratchblocks lines (4-space indented for nested
control structures). The `key` field is preserved through the pipeline so downstream files
can be joined back to the source examples.

## Quick start (inference only)

Reproduce sampling + SAC reranking on the SAC-primary 800 subset with the released Qwen adapter:

```bash
python3 src/inference/generate_predictions_select.py \
  --test data/test_sac_primary_subset_800.jsonl \
  --model satechi/qwen2.5-7b-sft-final-lr5e5-nl2scratch \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --out runs/qwen_sample.jsonl \
  --decoding sampling --num-candidates 8 --temperature 0.7 --top-p 0.9 --seed 42 \
  --parser src/inference/vm_pseudocode_parser.mjs

# All five selection rules from a single generation pass:
for FIELD in prediction prediction_likelihood prediction_reranked prediction_parse_likelihood prediction_parse_sac; do
  python3 src/inference/evaluate_predictions.py \
    --pred-file runs/qwen_sample.jsonl --prediction-field "${FIELD}" --postprocess
done
```

`--decoding greedy` instead gives the MAP/greedy baseline used as the table anchor.

## Training

### Causal LMs (Qwen / Llama)

Both use the same `train_causal_sft.py`. The Qwen run uses 4-bit QLoRA (the bf16-LoRA path is numerically unstable on Qwen2.5) and the Llama run uses bf16 LoRA. LoRA targets `{q,k,v,o,gate,up,down}_proj` in both cases.

The launchers auto-download the dataset on first run (see *Data*); pass
`HF_TOKEN` only when the *base* model is gated.

```bash
# Qwen
HF_TOKEN=hf_xxx bash scripts/sft/qwen_sft.sh

# Llama
HF_TOKEN=hf_xxx bash scripts/sft/llama_sft.sh
```

### FLAN-T5

```bash
bash scripts/sft/flan_sft.sh
```

## Evaluation & candidate selection

Two decoding regimes (greedy vs. sampling) and five selection rules over the sampled candidate set:

| Rule                    | Description                                                    |
|-------------------------|----------------------------------------------------------------|
| `first`                 | First decoded candidate (random pick under unbiased sampling). |
| `likelihood`            | Best-of-N by length-normalized log-likelihood.                 |
| `SAC`                   | argmax_i SAC_nl(x, y_i).                                       |
| `parse → likelihood`    | Parse-valid subset → likelihood argmax (fallback: full set).   |
| `parse → SAC`           | Parse-valid subset → SAC argmax (fallback: full set).          |

```bash
# Greedy baseline (Qwen)
MODEL=satechi/qwen2.5-7b-sft-final-lr5e5-nl2scratch \
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct  HF_TOKEN=hf_xxx \
  bash scripts/eval/eval_causal_greedy.sh

# Sampling + all five selection rules (Qwen)
USE_PARSER=1 \
MODEL=satechi/qwen2.5-7b-sft-final-lr5e5-nl2scratch \
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct  HF_TOKEN=hf_xxx \
  bash scripts/eval/eval_causal_sampling.sh

# Llama (same scripts, swap MODEL / BASE_MODEL)
USE_PARSER=1 \
MODEL=runs/checkpoints/llama-3.1-8b-instruct-sft \
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct  HF_TOKEN=hf_xxx \
  bash scripts/eval/eval_causal_sampling.sh

# FLAN-T5
MODEL=runs/checkpoints/flan-t5-base-sft \
  bash scripts/eval/eval_t5_sampling.sh
```

All metrics — exact match, line-exact match, token F1, parser pass rate, SAC_nl, SAC_pseudo (avg / =100% / ≥85%) — are written to plain-text files under `${OUTPUT_ROOT}/eval_outputs/`.

## Running Prompting Baselines

Prompting baselines live under `src/prompting_baseline/`. The main entrypoint is
`run_icl.py`, which retrieves few-shot examples from a local train JSONL file with
a lightweight TF-IDF index, calls an OpenAI chat model, and writes JSONL predictions.

The bundled diagnostic set can be used directly as the evaluation input:

```bash
export OPENAI_API_KEY=sk-...

python3 src/prompting_baseline/run_icl.py \
  --mode batch_submit \
  --train_file data/splits/train.jsonl \
  --test_file data/test_sac_primary_subset_800.jsonl \
  --output_file src/prompting_baseline/output/gpt54_diag800_20shot.jsonl \
  --model gpt-5.4 \
  --nshot 20
```

`data/splits/train.jsonl` is not bundled in this repository. Export the train split
from [`Heejindo/nl2scratch`](https://huggingface.co/datasets/Heejindo/nl2scratch)
to that path, or pass `--train_file` to another local JSONL file with the same
`key` / `nl` / `pseudocode` schema.

The prompting runner supports four modes:

| Mode            | Use case |
|-----------------|----------|
| `batch_submit`  | Create Batch API input, upload it, and print a `batch_id`. |
| `batch_collect` | Download a completed batch and write final predictions. |
| `batch_run`     | Submit, poll, and collect in one command. |
| `sync`          | Run synchronous API calls; useful for small debug runs with `--max_samples`. |

For asynchronous batches, collect results after the batch completes:

```bash
python3 src/prompting_baseline/run_icl.py \
  --mode batch_collect \
  --batch_id batch_xxx \
  --output_file src/prompting_baseline/output/gpt54_diag800_20shot.jsonl
```

For quick debugging:

```bash
python3 src/prompting_baseline/run_icl.py \
  --mode sync \
  --train_file data/splits/train.jsonl \
  --test_file data/test_sac_primary_subset_800.jsonl \
  --output_file src/prompting_baseline/output/debug_5shot.jsonl \
  --model gpt-5.4 \
  --nshot 5 \
  --max_samples 5
```

Prompting output rows use `predicted_pseudocode` and `gold_pseudocode`. Evaluate them
with the shared metrics script by selecting the prediction field:

```bash
python3 src/inference/evaluate_predictions.py \
  --pred-file src/prompting_baseline/output/gpt54_diag800_20shot.jsonl \
  --prediction-field predicted_pseudocode \
  --postprocess \
  --parser src/inference/vm_pseudocode_parser.mjs
```

Previously generated prompting-baseline outputs are kept in
`src/prompting_baseline/results/`.


## Hardware and runtime notes

The bash launchers are intended for **a single Linux machine with one CUDA-capable GPU**. Training Qwen-7B (4-bit QLoRA) or Llama-8B (bf16 LoRA) at the default `--max-seq-length 768` needs roughly **40 GB of VRAM**; FLAN-T5-base fits comfortably in 24 GB. Inference (greedy or N=8 sampling) is comparable in memory but faster.

For SLURM clusters, wrap a launcher in a thin sbatch file:

```bash
#!/bin/bash
#SBATCH ... your cluster's directives ...
bash scripts/sft/qwen_sft.sh
```

`HF_TOKEN`: pass it via the environment when launching.

## Citation

```bibtex
@misc{do2026nl2scratche,
      title={NL2Scratch: An Executable Benchmark and Evaluation for Block-Based Programming}, 
      author={Heejin Do and Alexandre Ballenghien and Yang Wu and April Yi Wang},
      year={2026},
      eprint={2606.22061},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.22061}, 
}
```
