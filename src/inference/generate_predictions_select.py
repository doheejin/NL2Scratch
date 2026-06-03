#!/usr/bin/env python3
"""Inference candidate selection over sampled (or greedy) candidates.

Generates N candidates per NL via nucleus sampling (the literature standard for
Best-of-N / MBR-style reranking) and emits the per-candidate breakdown plus
several selection-rule outputs in a single pass:

- prediction:              first candidate (decoding-order top-1; under unbiased
                           sampling this is a random pick over the candidate set).
- prediction_likelihood:   argmax over per-candidate generation log-probability
                           (Best-of-N by likelihood — strong "no semantic
                           reranker" baseline).
- prediction_reranked:     argmax over SAC alignment with the input NL
                           (= SAC rerank; kept under this name for back-compat
                           with the existing evaluator workflow).
- prediction_parse_likelihood: parse-valid subset → argmax likelihood
                           (structurally constrained Best-of-N; parse-only
                           signal). Only emitted when --parser is given.
                           Falls back to likelihood argmax over the full set
                           if no candidate parses.
- prediction_parse_sac:    parse-valid subset → argmax SAC (structural+semantic
                           multi-signal). Only emitted when --parser is given.
                           Falls back to the SAC argmax over the full set if no
                           candidate parses.

Per-candidate fields (text, logprob, sac_score, parse_ok) are saved under
'candidates' so downstream analysis can replay any other selection rule.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SRC_DIR))
from sac.semantic_alignment_check import (  # noqa: E402
    compute_slot_scores,
    extract_nl_slots,
    extract_pseudocode_slots,
)


SYSTEM_PROMPT = (
    "You convert natural-language Scratch instructions into valid Scratch pseudocode. "
    "Output only pseudocode. Use one block per line. Use exactly 4 leading spaces for nested "
    "blocks inside control structures. Preserve names, messages, numbers, and ordering."
)


def load_jsonl(path: Path, limit: int | None = None):
    with path.open() as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def build_prompt(nl: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{nl.strip()}\n"
        "<|assistant|>\n"
    )


def _autoload_tokenizer(name_or_path: str):
    try:
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=True, trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=False, trust_remote_code=True)


def _resolve_base_model(recorded: str | None, override: str | None) -> str:
    if override:
        return override
    if recorded and (not recorded.startswith("/") or Path(recorded).exists()):
        return recorded
    raise FileNotFoundError(
        f"PEFT adapter records base_model_name_or_path={recorded!r}, which is not a HF "
        "repo id and does not exist locally. Pass --base-model to override."
    )


def load_model_and_tokenizer(model_path: str, base_model_override: str | None = None):
    device_map = {"": 0} if torch.cuda.is_available() else None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    is_adapter = False
    peft_cfg = None
    try:
        peft_cfg = PeftConfig.from_pretrained(model_path)
        is_adapter = True
    except Exception:
        is_adapter = False

    if is_adapter:
        base_name = _resolve_base_model(peft_cfg.base_model_name_or_path, base_model_override)
        print(
            json.dumps({"stage": "resolved_base_model", "recorded": peft_cfg.base_model_name_or_path, "using": base_name}),
            flush=True,
        )
        tokenizer = _autoload_tokenizer(base_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=False)
    else:
        tokenizer = _autoload_tokenizer(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def collect_eos_token_ids(tokenizer) -> list[int]:
    """All chat/end-of-turn token IDs the tokenizer knows.

    transformers' generate() accepts a list for eos_token_id and stops on any of them.
    Critical for Llama-3.1: tokenizer.eos_token_id is <|end_of_text|> but the actual
    chat terminator is <|eot_id|>, so without this the model runs past turn-end into
    a repetition loop.
    """
    candidates = [
        "<|eot_id|>",            # Llama-3.1 / 3.2 chat end-of-turn
        "<|end_of_text|>",       # Llama-3.1 base EOS
        "<|im_end|>",            # Qwen / ChatML
        "<|endoftext|>",         # GPT-style
    ]
    ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    for tok in candidates:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid is not None and tid != tokenizer.unk_token_id and tid not in ids:
            ids.append(tid)
    return ids


def strip_prompt(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    marker = "<|assistant|>"
    if marker in full_text:
        return full_text.split(marker, 1)[1].strip()
    return full_text.strip()


def truncate_chat_continuation(text: str) -> str:
    trimmed = text or ""
    markers = (
        "<|end_of_text|>",
        "<|begin_of_text|>",
        "<|endoftext|>",
        "<|eot_id|>",
        "<|im_end|>",
        "<|im_start|>",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "Human:",
        "Assistant:",
    )
    for marker in markers:
        if marker in trimmed:
            trimmed = trimmed.split(marker, 1)[0].strip()
    return trimmed.strip()


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


def parse_ok(parser_path: Path, code: str) -> bool:
    if not code.strip():
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


def per_sequence_logprobs(model, sequences, scores, beam_indices=None):
    """Sum of log p(token_t | ...) over generated tokens, masking padding (-inf)."""
    transition = model.compute_transition_scores(
        sequences, scores, beam_indices=beam_indices, normalize_logits=True
    )
    # transition: (N, num_new_tokens); padding positions are -inf
    mask = transition != float("-inf")
    logprobs = transition.masked_fill(~mask, 0.0).sum(dim=-1)
    # Optional: length-normalize for fairness across variable-length candidates.
    lengths = mask.sum(dim=-1).clamp(min=1)
    return logprobs.tolist(), (logprobs / lengths).tolist()


def select_first(cands):
    return 0


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
    ap = argparse.ArgumentParser(description="Sampling-based candidate generation with multi-signal selection.")
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--model", required=True, help="HF repo id or local path (LoRA adapter or merged model).")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=384)
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
    model, tokenizer = load_model_and_tokenizer(args.model, base_model_override=args.base_model)
    model.eval()
    eos_token_ids = collect_eos_token_ids(tokenizer)
    print(json.dumps({"stage": "model_loaded", "eos_token_ids": eos_token_ids}), flush=True)
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
                    # Truncated trailing line from a prior crash; ignore.
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
            prompt = build_prompt(nl)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_len = inputs.input_ids.shape[1]

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "num_return_sequences": n_cand,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": eos_token_ids,
                "output_scores": True,
                "return_dict_in_generate": True,
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
                gen_out = model.generate(**inputs, **gen_kwargs)

            sequences = gen_out.sequences
            logprobs_sum, logprobs_avg = per_sequence_logprobs(model, sequences, gen_out.scores)

            cands: list[dict] = []
            for i, seq in enumerate(sequences):
                decoded = tokenizer.decode(seq, skip_special_tokens=True)
                text = truncate_chat_continuation(strip_prompt(decoded, prompt))
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
