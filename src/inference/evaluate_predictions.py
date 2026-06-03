#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SRC_DIR))
from sac.semantic_alignment_check import (  # noqa: E402
    SLOT_ORDER,
    compute_slot_scores,
    extract_nl_slots,
    extract_pseudocode_slots,
    scalar_similarity,
    set_similarity,
)


def _sac_average(scores):
    comparable = [v for v in scores if v is not None]
    return (sum(comparable) / len(comparable)) if comparable else 0.0


def compute_sac_nl(pred_text: str, nl: str) -> float:
    """SAC alignment between predicted pseudocode and NL (mirrors reranker scorer)."""
    lines = [ln for ln in (pred_text or "").splitlines() if ln.strip()]
    if not lines:
        return 0.0
    pseudo_slots = extract_pseudocode_slots(lines)
    nl_slots = extract_nl_slots(nl)
    slot_scores = compute_slot_scores(pseudo_slots, nl_slots, nl)
    return _sac_average(slot_scores.values())


def compute_sac_pseudo(pred_text: str, gold_text: str) -> float:
    """SAC alignment between predicted pseudocode and gold pseudocode.

    Uses plain set/scalar similarity (no NL-guided fuzzy matching), and honors the same
    loop_types/loop_counts exclusion as compute_slot_scores in §3.4.
    """
    pred_lines = [ln for ln in (pred_text or "").splitlines() if ln.strip()]
    gold_lines = [ln for ln in (gold_text or "").splitlines() if ln.strip()]
    if not pred_lines or not gold_lines:
        return 0.0
    pred_slots = extract_pseudocode_slots(pred_lines)
    gold_slots = extract_pseudocode_slots(gold_lines)
    scores: list[float | None] = []
    scores.append(scalar_similarity(pred_slots["event_type"], gold_slots["event_type"]))
    scores.append(scalar_similarity(pred_slots["event_target"], gold_slots["event_target"]))
    for slot in SLOT_ORDER[2:]:
        if slot in {"loop_types", "loop_counts"}:
            continue
        scores.append(set_similarity(set(pred_slots[slot]), set(gold_slots[slot])))
    return _sac_average(scores)


def normalize_lines(lines):
    return [" ".join(line.strip().split()) for line in lines if line.strip()]


def normalize_text(text: str):
    return "\n".join(normalize_lines(text.splitlines()))


def token_f1(pred: str, gold: str):
    p = pred.split()
    g = gold.split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    p_counts = {}
    g_counts = {}
    for t in p:
        p_counts[t] = p_counts.get(t, 0) + 1
    for t in g:
        g_counts[t] = g_counts.get(t, 0) + 1
    overlap = 0
    for t, c in p_counts.items():
        overlap += min(c, g_counts.get(t, 0))
    precision = overlap / max(1, len(p))
    recall = overlap / max(1, len(g))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


_BLOCK_KEYWORDS = (
    "when|move|turn|say|think|wait|broadcast|forever|repeat|if|else|end|set|change|"
    "go|glide|point|show|hide|play|stop|ask|switch|next|clear|pen|stamp|add|delete|"
    "replace|insert"
)

# Chat/special tokens that occasionally leak into decoded text (esp. Llama-3.1's
# <|end_of_text|> / <|begin_of_text|> when EOS handling misfires). Anything from
# the first marker onward is generation trash and must be cut.
_SPECIAL_TOKEN_MARKERS = (
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
)


def _strip_special_tokens(text: str) -> str:
    for m in _SPECIAL_TOKEN_MARKERS:
        if m in text:
            text = text.split(m, 1)[0]
    return text


def _truncate_block_repetition(lines: list[str], min_block: int = 5) -> list[str]:
    """Cut a 2x verbatim repeat of any block of >=min_block lines (anywhere in the output).

    Catches decoding degeneration where the same multi-line program fragment is emitted
    twice in a row past the natural stop. min_block=5 was tuned on Llama/Qwen sampling
    outputs to give a large Llama win with no measurable Qwen regression.
    """
    n = len(lines)
    for k in range(min_block, n // 2 + 1):
        for i in range(0, n - 2 * k + 1):
            if lines[i:i + k] == lines[i + k:i + 2 * k]:
                return lines[:i + k]
    return lines


def _fix_unclosed_blocks(lines: list[str]) -> list[str]:
    stack = []

    def _is_block_start(line: str) -> str | None:
        lower = line.lower()
        if lower.startswith("repeat "):
            return "repeat"
        if lower == "forever":
            return "forever"
        if lower.startswith("if ") and lower.endswith("then"):
            return "if"
        return None

    cleaned = []
    for line in lines:
        lower = line.lower()
        if lower == "end":
            if stack:
                stack.pop()
                cleaned.append(line)
            continue
        if lower == "else":
            if stack and stack[-1]["type"] == "if":
                cleaned.append(line)
                continue
            continue

        block_type = _is_block_start(line)
        if block_type:
            if block_type == "forever" and any(ctx["type"] == "forever" for ctx in stack):
                continue
            stack.append({"type": block_type})
            cleaned.append(line)
            continue

        cleaned.append(line)

    while stack:
        stack.pop()
        cleaned.append("end")

    filtered = []
    i = 0
    while i < len(cleaned):
        if cleaned[i].lower() == "forever" and i + 1 < len(cleaned) and cleaned[i + 1].lower() == "end":
            i += 2
            continue
        filtered.append(cleaned[i])
        i += 1

    return filtered


def postprocess_pred(text: str) -> str:
    s = _strip_special_tokens(text or "").strip()
    if not s:
        return s
    if "\n" not in s:
        s = re.sub(rf"\s+(?=({_BLOCK_KEYWORDS})\b)", "\n", s, flags=re.IGNORECASE)
    lines = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(and|then)\s+", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+(and|then)$", "", line, flags=re.IGNORECASE)
        if re.match(r"^when\s+green\s+flag\s+clicked\b", line, flags=re.IGNORECASE):
            line = re.sub(r"^when\s+green\s+flag\s+clicked\b", "when @greenFlag clicked", line, flags=re.IGNORECASE)
        lowered = line.lower()
        if lowered.startswith("forever:"):
            line = "forever"
        if lowered.startswith("wait until ") and not lowered.startswith("wait until <"):
            cond = line[len("wait until "):].strip()
            cond = cond.rstrip(">")
            line = f"wait until <{cond}>"
        if lowered.startswith("if touching") and "edge" in lowered:
            line = "if <touching (edge v)?> then"
        else:
            m_touch = re.match(r"^if touching\s+\[?([^\]\?]+)\]?\??>", line, flags=re.IGNORECASE)
            if not m_touch:
                m_touch = re.match(r"^if touching\s+([A-Za-z0-9_ ]+)\??", line, flags=re.IGNORECASE)
            if m_touch:
                target = m_touch.group(1).strip()
                line = f"if <touching ({target} v)?> then"
        if lowered.startswith("if ") and " then" not in lowered:
            cond = line[3:].strip()
            if not cond.startswith("<"):
                cond = f"<{cond}>"
            if not cond.endswith(">"):
                cond = cond + ">"
            line = f"if {cond} then"
        lines.append(line)
    lines = _fix_unclosed_blocks(lines)
    lines = _truncate_block_repetition(lines, min_block=5)
    return "\n".join(lines).strip()


def vm_parse_ok(parser_path: Path, code: str):
    res = subprocess.run(
        ["node", str(parser_path)],
        input=code.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return res.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-file", type=Path, required=True)
    ap.add_argument("--parser", type=Path, default=None)
    ap.add_argument("--postprocess", action="store_true")
    ap.add_argument("--print-mismatch", action="store_true")
    ap.add_argument("--print-limit", type=int, default=5)
    ap.add_argument(
        "--prediction-field",
        default="prediction",
        help="JSON field to score. Use 'prediction_reranked' to score the SAC-reranked output.",
    )
    args = ap.parse_args()

    exact = 0
    line_exact = 0
    f1_sum = 0.0
    parse_ok = 0
    total = 0
    mismatch_printed = 0
    sac_nl_sum = 0.0
    sac_nl_perfect = 0
    sac_nl_high = 0
    sac_pseudo_sum = 0.0
    sac_pseudo_perfect = 0
    sac_pseudo_high = 0

    for line in args.pred_file.open():
        if not line.strip():
            continue
        obj = json.loads(line)
        pred = obj.get(args.prediction_field, "")
        gold = obj.get("gold", "")
        nl = obj.get("nl", "")
        if isinstance(gold, list):
            gold = "\n".join(gold)
        pred_eval = postprocess_pred(pred) if args.postprocess else pred
        pred_norm = normalize_text(pred_eval)
        gold_norm = normalize_text(gold)
        pred_norm = re.sub(r"[，,:]", " ", pred_norm)
        gold_norm = re.sub(r"[，,:]", " ", gold_norm)
        pred_norm = pred_norm.encode("ascii", "ignore").decode()
        gold_norm = gold_norm.encode("ascii", "ignore").decode()
        if pred_norm == gold_norm:
            exact += 1
        if normalize_lines(pred_eval.splitlines()) == normalize_lines(gold.splitlines()):
            line_exact += 1
        f1_sum += token_f1(pred_norm, gold_norm)
        if args.parser:
            if vm_parse_ok(args.parser, pred_norm):
                parse_ok += 1

        sac_nl = compute_sac_nl(pred_eval, nl)
        sac_pseudo = compute_sac_pseudo(pred_eval, gold)
        sac_nl_sum += sac_nl
        sac_pseudo_sum += sac_pseudo
        if sac_nl >= 1.0:
            sac_nl_perfect += 1
        if sac_nl >= 0.85:
            sac_nl_high += 1
        if sac_pseudo >= 1.0:
            sac_pseudo_perfect += 1
        if sac_pseudo >= 0.85:
            sac_pseudo_high += 1

        if args.print_mismatch and mismatch_printed < args.print_limit and pred_norm != gold_norm:
            print("--- mismatch ---")
            print("NL:", nl)
            print("PRED:", pred_eval)
            print("GOLD:", gold)
            mismatch_printed += 1
        total += 1

    denom = max(1, total)
    print(f"examples: {total}")
    print(f"exact_match: {exact / denom:.4f}")
    print(f"line_exact_match: {line_exact / denom:.4f}")
    print(f"token_f1: {f1_sum / denom:.4f}")
    if args.parser:
        print(f"vm_parse_rate: {parse_ok / denom:.4f}")
    print(f"sac_nl_avg: {sac_nl_sum / denom:.4f}")
    print(f"sac_nl_perfect_rate: {sac_nl_perfect / denom:.4f}")
    print(f"sac_nl_high_rate: {sac_nl_high / denom:.4f}")
    print(f"sac_pseudo_avg: {sac_pseudo_sum / denom:.4f}")
    print(f"sac_pseudo_perfect_rate: {sac_pseudo_perfect / denom:.4f}")
    print(f"sac_pseudo_high_rate: {sac_pseudo_high / denom:.4f}")


if __name__ == "__main__":
    main()
