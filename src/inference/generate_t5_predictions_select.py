#!/usr/bin/env python3
"""Sampling-based candidate generation with multi-signal selection for Flan-T5.

Encoder-decoder counterpart of src/inference/generate_predictions_select.py.
Same per-record output schema (prediction, prediction_likelihood, prediction_reranked,
[prediction_parse_likelihood, prediction_parse_sac], candidates[...]) so the existing
evaluator (src/inference/evaluate_predictions.py) can score any field.

Key differences from the causal-LM version:
- Uses AutoModelForSeq2SeqLM (T5). No PEFT adapter logic; the checkpoint is a full fine-tune.
- No chat prompt: T5 takes raw NL as input (matches the original eval pipeline).
- No prompt-prefix stripping: generate() returns only decoded target tokens.
- No chat-token EOS list: T5's tokenizer.eos_token_id is sufficient.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SRC_DIR))
from sac.semantic_alignment_check import (  # noqa: E402
    compute_slot_scores,
    extract_nl_slots,
    extract_pseudocode_slots,
)


def load_jsonl(path: Path, limit: int | None = None):
    with path.open() as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def sac_alignment_score(nl: str, pseudocode_text: str) -> float:
    if not pseudocode_text.strip():
        return 0.0
    lines = [ln for ln in pseudocode_text.splitlines() if ln.strip()]
    try:
        pseudo_slots = extract_pseudocode_slots(lines)
        nl_slots = extract_nl_slots(nl)
        slot_scores = compute_slot_scores(pseudo_slots, nl_slots, nl)
    except Exception:
        return 0.0
    comparable = [v for v in slot_scores.values() if v is not None]
    if not comparable:
        return 0.0
    return sum(comparable) / len(comparable)


def parse_ok(parser_path: Path | None, code: str) -> bool:
    if parser_path is None or not code.strip():
        return False
    try:
        res = subprocess.run(
            ["node", str(parser_path)],
            input=code.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=20,
        )
        return res.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def per_sequence_logprobs(model, sequences, scores):
    """Sum and mean of log p(token_t | ...) over generated tokens, masking padding."""
    transition = model.compute_transition_scores(sequences, scores, normalize_logits=True)
    mask = transition != float("-inf")
    logprobs = transition.masked_fill(~mask, 0.0).sum(dim=-1)
    lengths = mask.sum(dim=-1).clamp(min=1)
    return logprobs.tolist(), (logprobs / lengths).tolist()


def select_argmax(cands, key):
    best_i, best_v = 0, key(cands[0])
    for i in range(1, len(cands)):
        v = key(cands[i])
        if v > best_v:
            best_v, best_i = v, i
    return best_i


def _argmax_in_pool(cands, pool, key):
    best_i, best_v = pool[0], key(cands[pool[0]])
    for i in pool[1:]:
        v = key(cands[i])
        if v > best_v:
            best_v, best_i = v, i
    return best_i


def select_parse_then(cands, key):
    """Parse-valid subset → argmax key. Fall back to full set when none parse."""
    valid = [i for i, c in enumerate(cands) if c.get("parse_ok") is True]
    pool = valid if valid else list(range(len(cands)))
    return _argmax_in_pool(cands, pool, key)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sampling-based candidate generation with multi-signal selection (Flan-T5).")
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--model", required=True, help="HF repo id or local path of a Seq2Seq LM (T5 family).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-source", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--decoding",
        choices=["sampling", "greedy"],
        default="sampling",
        help="sampling: do_sample=True with temperature/top-p; greedy: deterministic single candidate.",
    )
    ap.add_argument("--num-candidates", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--parser",
        type=Path,
        default=None,
        help="Path to vm_pseudocode_parser.mjs to compute per-candidate parse_ok; enables prediction_parse_sac.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If --out exists, skip rows whose key already appears in it and append remaining results.",
    )
    args = ap.parse_args()

    if args.decoding == "greedy":
        n_cand = 1
    else:
        n_cand = args.num_candidates

    if args.parser is not None and not args.parser.exists():
        raise FileNotFoundError(f"--parser path does not exist: {args.parser}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # T5 base in bf16 is generally safe on modern GPUs; fall back to fp32 on CPU.
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(
        json.dumps(
            {
                "stage": "loading_model",
                "model": args.model,
                "decoding": args.decoding,
                "n_cand": n_cand,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "parser": str(args.parser) if args.parser else None,
            }
        ),
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    print(json.dumps({"stage": "model_loaded"}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[str] = set()
    open_mode = "w"
    if args.resume and args.out.exists():
        with args.out.open("r") as in_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = rec.get("key")
                if k:
                    done_keys.add(k)
        open_mode = "a"
        print(json.dumps({"stage": "resume", "already_done": len(done_keys)}), flush=True)

    with args.out.open(open_mode) as out_f:
        for idx, row in enumerate(load_jsonl(args.test, limit=args.limit), start=1):
            key = row.get("key", "")
            if key and key in done_keys:
                print(json.dumps({"stage": "skip_done", "idx": idx, "key": key}), flush=True)
                continue
            nl = row["nl"]
            print(json.dumps({"stage": "start_example", "idx": idx, "key": key}), flush=True)

            enc = tokenizer(
                nl,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_source,
            ).to(device)

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "num_return_sequences": n_cand,
                "output_scores": True,
                "return_dict_in_generate": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.decoding == "greedy":
                gen_kwargs.update({"do_sample": False, "num_beams": 1})
            else:
                gen_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "num_beams": 1,
                    }
                )

            with torch.no_grad():
                gen_out = model.generate(**enc, **gen_kwargs)

            sequences = gen_out.sequences
            logprobs_sum, logprobs_avg = per_sequence_logprobs(model, sequences, gen_out.scores)

            cands: list[dict] = []
            for i, seq in enumerate(sequences):
                # T5 returns only the decoded target; skip_special_tokens drops <pad>/</s>.
                text = tokenizer.decode(seq, skip_special_tokens=True).strip()
                cands.append(
                    {
                        "text": text,
                        "logprob_sum": logprobs_sum[i],
                        "logprob_mean": logprobs_avg[i],
                        "sac_score": sac_alignment_score(nl, text),
                        "parse_ok": parse_ok(args.parser, text) if args.parser is not None else None,
                    }
                )

            i_first = 0
            i_lik = select_argmax(cands, key=lambda c: c["logprob_mean"])
            i_sac = select_argmax(cands, key=lambda c: c["sac_score"])
            record = {
                "key": key,
                "nl": nl,
                "gold": row.get("pseudocode", ""),
                "num_candidates": len(cands),
                "decoding": args.decoding,
                "prediction": cands[i_first]["text"],
                "prediction_likelihood": cands[i_lik]["text"],
                "prediction_reranked": cands[i_sac]["text"],
                "sac_score": cands[i_sac]["sac_score"],
                "sac_score_top1": cands[i_first]["sac_score"],
                "best_sac_idx": i_sac,
                "best_likelihood_idx": i_lik,
                "candidates": cands,
            }
            if args.parser is not None:
                i_pl = select_parse_then(cands, key=lambda c: c["logprob_mean"])
                i_ps = select_parse_then(cands, key=lambda c: c["sac_score"])
                record["prediction_parse_likelihood"] = cands[i_pl]["text"]
                record["prediction_parse_sac"] = cands[i_ps]["text"]
                record["best_parse_likelihood_idx"] = i_pl
                record["best_parse_sac_idx"] = i_ps
                record["any_parse_ok"] = any(c["parse_ok"] for c in cands)

            out_f.write(json.dumps(record, ensure_ascii=True) + "\n")
            out_f.flush()
            print(
                json.dumps(
                    {
                        "stage": "wrote_example",
                        "idx": idx,
                        "key": key,
                        "sac_score": round(cands[i_sac]["sac_score"], 4),
                        "sac_score_top1": round(cands[i_first]["sac_score"], 4),
                        "best_sac_idx": i_sac,
                        "best_likelihood_idx": i_lik,
                        "any_parse_ok": record.get("any_parse_ok"),
                    }
                ),
                flush=True,
            )

    print(json.dumps({"wrote": str(args.out)}))


if __name__ == "__main__":
    main()
