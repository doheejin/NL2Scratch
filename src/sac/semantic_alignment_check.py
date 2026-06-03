#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}

NON_ASCII_WORD_RE = r"[^\W\d_][^\W_]*"

SLOT_ORDER = [
    "event_type",
    "event_target",
    "action_types",
    "loop_types",
    "loop_counts",
    "condition_types",
    "numeric_values",
    "variable_names",
    "costume_names",
    "backdrop_names",
    "broadcast_names",
]

EVENT_LEXICON = {
    "green_flag": ["green flag"],
    "backdrop_switch": ["backdrop", "background"],
    "receive": ["receive"],
    "key_pressed": ["key", "press", "pressed", "pressing", "hold", "holding", "let go"],
    "sprite_clicked": ["sprite", "click"],
}

BACKDROP_SWITCH_VERBS = ["change", "changes", "changed", "switch", "switches", "switching"]
LOOP_FOREVER_CUES = ["forever", "over and over", "again and again"]
LOOP_REPEAT_ONCE_CUES = ["just once", "one time"]

ACTION_LEXICON = {
    "show": ["show up", "shows up", "show me", "show it", "be visible", "is visible", "showing", "appear"],
    "hide": ["hide", "disappear", "invisible", "be hidden"],
    "glide": ["glide"],
    "go_to": [
        "move to the spot",
        "moving to the spot",
        "move it to the spot",
        "move me to ",
        "move it to ",
        "move to x",
        "go to x",
        "go to the spot",
        "go to ",
        "random spot",
        "middle of the screen",
    ],
    "move": ["move", "moving", "forward", "backward", "backwards"],
    "turn_left": ["turn left", "rotate left", "counterclockwise"],
    "turn_right": ["turn right", "rotate right", "clockwise"],
    "point_direction": [
        "point in direction",
        "point left",
        "point to the left",
        "point right",
        "point to the right",
        "face to the right",
        "facing to the right",
        "face to the left",
        "facing to the left",
    ],
    "point_towards": ["point towards"],
    "say": ["say "],
    "think": ["think "],
    "wait": ["wait"],
    "switch_costume": [
        "change to costume",
        "change to ",
        "change the costume to",
        "change your costume to",
        "change its costume to",
        "change my costume to",
        "change costume to",
        "switch to the",
        "switch to costume",
        "switch costume to",
        "switch your costume to",
        "switch its costume to",
        "switch my costume to",
    ],
    "switch_backdrop": ["change the backdrop to", "switch backdrop to"],
    "next_costume": ["next costume"],
    "next_backdrop": ["next backdrop"],
    "set_variable": ["set ", "make ", "update "],
    "change_variable": [
        "change ",
        "subtract",
        "add ",
        "take away",
        "decrease",
        "reduce",
        "lower ",
        "move up by",
        "moving up by",
        "move down by",
        "moving down by",
        "move left by",
        "moving left by",
        "move right by",
        "moving right by",
        "go down by",
        "go up by",
    ],
    "play_sound": ["play sound", "play the sound"],
    "stop_sounds": ["stop all sounds"],
    "broadcast": ["broadcast", "send message", "send out the message", "send out message"],
    "ask": ["ask "],
}

CONDITION_LEXICON = {
    "touching": ["touching", "touch "],
    "edge": ["edge"],
    "key_pressed": ["pressed", "pressing", "hold", "holding", "let go"],
    "mouse_down": ["mouse down"],
    "random": ["random number", "random"],
    "answer": ["answer"],
    "greater_than": ["greater than", "more than"],
    "less_than": ["less than", "smaller than"],
}

VARIABLE_DIRECTION_MAP = {
    "x": ["move left by", "moving left by", "move right by", "moving right by"],
    "y": ["move up by", "moving up by", "move down by", "moving down by", "go up a little bit", "y position"],
}

UNICODE_WORD_RE = r"[^\W\d_][^\W_]*(?:[ .'\-][^\W\d_][^\W_]*)*"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_name(value: str) -> str:
    value = value.strip().lower()
    value = value.strip("\"' ")
    value = re.sub(r"\s+", " ", value)
    value = value.replace(" v", "")
    return value


def normalize_number(value: float | int | str) -> str:
    try:
        f = float(value)
    except (ValueError, OverflowError):
        return str(value)
    if not math.isfinite(f):
        # Out-of-range literals (e.g. model-hallucinated 200-digit numbers parse to inf).
        return str(value)
    if math.isclose(f, round(f), abs_tol=1e-9):
        return str(int(round(f)))
    return f"{f:.6f}".rstrip("0").rstrip(".")


def extract_digit_numbers(text: str) -> set[str]:
    return {normalize_number(match) for match in re.findall(r"-?\d+(?:\.\d+)?", text)}


def extract_number_words(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    out: set[str] = set()
    scales = {"hundred": 100, "thousand": 1000}
    i = 0
    while i < len(words):
        if words[i] not in NUMBER_WORDS and words[i] != "a":
            i += 1
            continue

        current = 0
        total = 0
        j = i
        consumed = False
        while j < len(words):
            word = words[j]
            if word == "a":
                current += 1
                consumed = True
                j += 1
                continue
            if word in NUMBER_WORDS and word not in scales:
                current += NUMBER_WORDS[word]
                consumed = True
                j += 1
                continue
            if word == "hundred":
                current = max(1, current) * 100
                consumed = True
                j += 1
                continue
            if word == "thousand":
                total += max(1, current) * 1000
                current = 0
                consumed = True
                j += 1
                continue
            break

        if consumed:
            value = total + current
            if value > 0:
                out.add(normalize_number(value))
            i = j
        else:
            i += 1
    return out


def extract_numbers(text: str) -> set[str]:
    return extract_digit_numbers(text) | extract_number_words(text)


def extract_repeat_counts(text: str) -> set[str]:
    counts: set[str] = set()
    patterns = [
        r"repeat ([a-z0-9 .-]+?) times",
        r"do this ([a-z0-9 .-]+?) times",
        r"do that ([a-z0-9 .-]+?) times",
        r"do this next thing ([a-z0-9 .-]+?) times",
        r"do all this ([a-z0-9 .-]+)",
        r"do that whole thing ([a-z0-9 .-]+?) times",
        r"do this whole thing ([a-z0-9 .-]+?) times",
        r"play [^,.]+ sound [^,.]*? ([a-z0-9 .-]+?) times",
        r"keep doing this ([a-z0-9 .-]+?) times",
        r"(?:move|turn|glide|wait|say|think|go)\b[^,.]{0,40}\b([a-z0-9 .-]+?) times",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            fragment = match.group(1).strip()
            counts.update(extract_numbers(fragment))
    return counts


def has_any_phrase(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def add_signed_directional_numbers(target: set[str], text: str) -> set[str]:
    signed_values: set[str] = set()
    patterns = [
        (r"(?:move|moving|go|glide)\s+back(?:ward)?s?\s+(-?\d+(?:\.\d+)?)\s+steps?", -1),
        (r"(?:move|moving|go|glide)\s+forward\s+(-?\d+(?:\.\d+)?)\s+steps?", 1),
        (r"(?:move|moving|go)\s+left\s+by\s+(-?\d+(?:\.\d+)?)", -1),
        (r"(?:move|moving|go)\s+right\s+by\s+(-?\d+(?:\.\d+)?)", 1),
        (r"(?:move|moving|go)\s+down\s+by\s+(-?\d+(?:\.\d+)?)", -1),
        (r"(?:move|moving|go)\s+up\s+by\s+(-?\d+(?:\.\d+)?)", 1),
        (r"(?:move|moving|go)\s+left\s+(-?\d+(?:\.\d+)?)\s+steps?", -1),
        (r"(?:move|moving|go)\s+right\s+(-?\d+(?:\.\d+)?)\s+steps?", 1),
        (r"(?:move|moving|go)\s+down\s+(-?\d+(?:\.\d+)?)\s+steps?", -1),
        (r"(?:move|moving|go)\s+up\s+(-?\d+(?:\.\d+)?)\s+steps?", 1),
    ]
    for pattern, sign in patterns:
        for raw in re.findall(pattern, text):
            value = float(raw)
            signed = normalize_number(sign * abs(value))
            target.add(signed)
            signed_values.add(signed)
    return signed_values


def extract_named_target(text: str, prefixes: list[str]) -> str | None:
    for prefix in prefixes:
        pattern = rf"{prefix}\s+([^,.]+?)(?:,|\.| and\b| then\b|$)"
        m = re.search(pattern, text)
        if m:
            return normalize_name(m.group(1))
    return None


def extract_costume_sequence_targets(text: str) -> set[str]:
    patterns = [
        r"(?:change the costume to|change (?:its |your |my )?costume to|change costume to)\s+([a-z0-9_\-]+)",
        r"(?:switch (?:its |your |my )?costume to|switch costume to)\s+([a-z0-9_\-]+)",
        r"(?:change it to|switch it to|change to|switch to)\s+([a-z0-9_\-]+)(?:\s+costume)?",
    ]
    out: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            name = normalize_name(match.group(1))
            if name and name not in {"the", "it"}:
                out.add(name)
    return out


def extract_sound_targets(text: str) -> set[str]:
    patterns = [
        r"play sound \(([^)]+?) v\)",
        r"play (?:the )?([a-z0-9_\-]+) sound",
        r"play the sound ([a-z0-9_\-]+)",
        r"play sound ([a-z0-9_\-]+)",
    ]
    out: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            name = normalize_name(match.group(1))
            if name:
                out.add(name)
    return out


def mask_identifier_numbers(text: str, identifiers: set[str]) -> str:
    masked = text
    for identifier in sorted({normalize_name(x) for x in identifiers if x}, key=len, reverse=True):
        if not identifier:
            continue
        pattern = re.escape(identifier)
        masked = re.sub(pattern, " ", masked, flags=re.IGNORECASE)
    return masked


def is_reasonable_variable_name(name: str) -> bool:
    name = normalize_name(name)
    if not name:
        return False
    bad_prefixes = [
        "it ",
        "the sprite ",
        "sprite ",
        "sure ",
        "if ",
        "when ",
    ]
    bad_tokens = {
        "switch",
        "glide",
        "wait",
        "show",
        "hide",
        "move",
        "turn",
        "point",
        "play",
        "ask",
        "say",
        "think",
        "click",
    }
    if any(name.startswith(prefix) for prefix in bad_prefixes):
        return False
    if any(token in name.split() for token in bad_tokens):
        return False
    return True


def extract_pseudocode_slots(pseudocode: list[str] | str) -> dict[str, Any]:
    lines = pseudocode if isinstance(pseudocode, list) else [pseudocode]
    text = "\n".join(lines)
    lower_text = text.lower()
    slots: dict[str, Any] = {
        "event_type": None,
        "event_target": None,
        "action_types": set(),
        "loop_types": set(),
        "loop_counts": set(),
        "condition_types": set(),
        "numeric_values": set(),
        "variable_names": set(),
        "costume_names": set(),
        "backdrop_names": set(),
        "broadcast_names": set(),
    }

    if lines:
        first = lines[0].strip().lower()
        if first.startswith("when @greenflag clicked"):
            slots["event_type"] = "green_flag"
        elif first.startswith("when backdrop switches to ["):
            slots["event_type"] = "backdrop_switch"
            m = re.search(r"when backdrop switches to \[([^\]]+)\]", first)
            if m:
                slots["event_target"] = normalize_name(m.group(1))
                slots["backdrop_names"].add(normalize_name(m.group(1)))
        elif first.startswith("when i receive ["):
            slots["event_type"] = "receive"
            m = re.search(r"when i receive \[([^\]]+)\]", first)
            if m:
                slots["event_target"] = normalize_name(m.group(1))
                slots["broadcast_names"].add(normalize_name(m.group(1)))
        elif " key pressed" in first:
            slots["event_type"] = "key_pressed"
            m = re.search(r"when [\[\(]([^\]\)]+?)(?: v)?[\]\)] key pressed", first)
            if m:
                slots["event_target"] = normalize_name(m.group(1))
        elif first.startswith("when this sprite clicked"):
            slots["event_type"] = "sprite_clicked"

    for line in lines:
        s = line.strip()
        lower = s.lower()

        if lower == "forever":
            slots["loop_types"].add("forever")
        if lower.startswith("repeat until"):
            slots["loop_types"].add("repeat_until")
        for repeat in re.findall(r"repeat \((-?\d+(?:\.\d+)?)\)", lower):
            slots["loop_types"].add("repeat")
            slots["loop_counts"].add(normalize_number(repeat))

        if lower.startswith("show"):
            slots["action_types"].add("show")
        if lower.startswith("hide"):
            slots["action_types"].add("hide")
        if lower.startswith("go to x:") or lower.startswith("go to ("):
            slots["action_types"].add("go_to")
        if lower.startswith("glide "):
            slots["action_types"].add("glide")
        if lower.startswith("move "):
            slots["action_types"].add("move")
        if lower.startswith("turn left"):
            slots["action_types"].add("turn_left")
        if lower.startswith("turn right"):
            slots["action_types"].add("turn_right")
        if lower.startswith("point in direction"):
            slots["action_types"].add("point_direction")
        if lower.startswith("point towards"):
            slots["action_types"].add("point_towards")
        if lower.startswith("say "):
            slots["action_types"].add("say")
        if lower.startswith("think "):
            slots["action_types"].add("think")
        if lower.startswith("wait "):
            slots["action_types"].add("wait")
        if lower.startswith("switch costume to"):
            slots["action_types"].add("switch_costume")
            m = re.search(r"switch costume to \[([^\]]+)\]", lower)
            if m:
                slots["costume_names"].add(normalize_name(m.group(1)))
        if lower.startswith("switch backdrop to"):
            slots["action_types"].add("switch_backdrop")
            m = re.search(r"switch backdrop to \[([^\]]+)\]", lower)
            if m:
                slots["backdrop_names"].add(normalize_name(m.group(1)))
        if lower.startswith("next costume"):
            slots["action_types"].add("next_costume")
        if lower.startswith("next backdrop"):
            slots["action_types"].add("next_backdrop")
        if lower.startswith("set "):
            slots["action_types"].add("set_variable")
        if lower.startswith("change "):
            slots["action_types"].add("change_variable")
            if lower.startswith("change x by") or lower.startswith("change y by"):
                slots["action_types"].add("move")
        if lower.startswith("broadcast "):
            slots["action_types"].add("broadcast")
            m = re.search(r"broadcast \(([^)]+)", lower)
            if m:
                slots["broadcast_names"].add(normalize_name(m.group(1)))
        if lower.startswith("play sound"):
            slots["action_types"].add("play_sound")
        if lower.startswith("stop all sounds"):
            slots["action_types"].add("stop_sounds")
        if lower.startswith("ask "):
            slots["action_types"].add("ask")

        for match in re.findall(r"(?:set|change) \[([^\]]+?) v\]", lower):
            slots["variable_names"].add(normalize_name(match))
        if lower.startswith("set x to") or lower.startswith("change x by"):
            slots["variable_names"].add("x")
        if lower.startswith("set y to") or lower.startswith("change y by"):
            slots["variable_names"].add("y")
        if lower.startswith("set size to") or lower.startswith("change size by"):
            slots["variable_names"].add("size")
        if lower.startswith("set volume to") or lower.startswith("change volume by"):
            slots["variable_names"].add("volume")

        if "if <" in lower or "wait until <" in lower or lower.startswith("repeat until"):
            if "touching" in lower:
                slots["condition_types"].add("touching")
            if "edge" in lower:
                slots["condition_types"].add("edge")
            if "key (" in lower and "pressed?" in lower:
                slots["condition_types"].add("key_pressed")
            if "mouse down?" in lower:
                slots["condition_types"].add("mouse_down")
            if "pick random" in lower:
                slots["condition_types"].add("random")
            if "answer" in lower:
                slots["condition_types"].add("answer")
            if "=" in lower:
                slots["condition_types"].add("equals")
            if " > " in lower or "> (" in lower:
                slots["condition_types"].add("greater_than")
            if " < " in lower or "< (" in lower:
                slots["condition_types"].add("less_than")

    identifier_names = set()
    if slots["event_target"]:
        identifier_names.add(slots["event_target"])
    for name_slot in ["variable_names", "costume_names", "backdrop_names", "broadcast_names"]:
        identifier_names.update(slots[name_slot])
    identifier_names.update(extract_sound_targets(lower_text))
    masked_text = mask_identifier_numbers(lower_text, identifier_names)
    slots["numeric_values"] = extract_numbers(masked_text)

    return slots


def extract_nl_slots(nl: str) -> dict[str, Any]:
    lower = nl.lower()
    start_clause = re.split(r",|\.|\bthen\b", lower, maxsplit=1)[0]
    trigger_clause = start_clause.strip()
    costume_targets = extract_costume_sequence_targets(lower)
    is_costume_or_backdrop_change = any(
        phrase in lower
        for phrase in [
            "change the costume to",
            "change your costume to",
            "change its costume to",
            "change my costume to",
            "change costume to",
            "change to costume",
            "switch costume to",
            "change the backdrop to",
            "change the background to",
            "switch backdrop to",
        ]
    )
    is_broadcast_phrase = "send out the message" in lower or "send message" in lower or "broadcast " in lower
    slots: dict[str, Any] = {
        "event_type": None,
        "event_target": None,
        "action_types": set(),
        "loop_types": set(),
        "loop_counts": set(),
        "condition_types": set(),
        "numeric_values": set(),
        "variable_names": set(),
        "costume_names": set(),
        "backdrop_names": set(),
        "broadcast_names": set(),
    }
    directional_values: set[str] = set()

    if "click on this sprite" in trigger_clause or "click this sprite" in trigger_clause or "click on the sprite" in trigger_clause:
        slots["event_type"] = "sprite_clicked"
    elif has_any_phrase(trigger_clause, EVENT_LEXICON["green_flag"]):
        slots["event_type"] = "green_flag"
    elif (
        has_any_phrase(trigger_clause, EVENT_LEXICON["backdrop_switch"])
        and has_any_phrase(trigger_clause, BACKDROP_SWITCH_VERBS)
    ):
        slots["event_type"] = "backdrop_switch"
        name = extract_named_target(
            trigger_clause,
            [
                r"when the backdrop changes to",
                r"when the background changes to",
                r"when we change the backdrop to",
                r"when we change the background to",
                r"backdrop changes to",
                r"background changes to",
                r"backdrop switches to",
            ],
        )
        if name:
            slots["event_target"] = name
            slots["backdrop_names"].add(name)
    elif has_any_phrase(lower, EVENT_LEXICON["receive"]):
        slots["event_type"] = "receive"
        m = re.search(r"receive ([a-z0-9_\- !?]+?)(?:,|\.| then| and|$)", lower)
        if m:
            name = normalize_name(m.group(1))
            slots["event_target"] = name
            slots["broadcast_names"].add(name)
    elif has_any_phrase(trigger_clause, EVENT_LEXICON["key_pressed"]):
        slots["event_type"] = "key_pressed"
        m = re.search(r'when (?:the )?(?:"([^"]+)"|([a-z0-9_ ]+?)) key (?:is )?pressed', trigger_clause)
        if not m:
            m = re.search(r'press(?:ed|ing)? (?:the )?(?:"([^"]+)"|([a-z0-9_ ]+?)) key', trigger_clause)
        if not m:
            m = re.search(r'hold(?:ing)? (?:the )?(?:"([^"]+)"|([a-z0-9_ ]+?)) key', trigger_clause)
        if not m:
            m = re.search(r'let go of (?:the )?(?:"([^"]+)"|([a-z0-9_ ]+?)) key', trigger_clause)
        if m:
            slots["event_target"] = normalize_name(next(g for g in m.groups() if g))
    elif has_any_phrase(trigger_clause, EVENT_LEXICON["sprite_clicked"]):
        slots["event_type"] = "sprite_clicked"

    if has_any_phrase(lower, LOOP_FOREVER_CUES):
        slots["loop_types"].add("forever")
    repeat_counts = extract_repeat_counts(lower)
    if repeat_counts:
        slots["loop_types"].add("repeat")
        slots["loop_counts"].update(repeat_counts)
    if has_any_phrase(lower, LOOP_REPEAT_ONCE_CUES):
        slots["loop_types"].add("repeat")
        slots["loop_counts"].add("1")
        slots["numeric_values"].add("1")
    if "twice" in lower or ("one more time" in lower and "do that again" in lower):
        slots["loop_types"].add("repeat")
        slots["loop_counts"].add("2")
        slots["numeric_values"].add("2")
    if "until" in lower:
        slots["loop_types"].add("repeat_until")
    elif "as long as" in lower:
        slots["loop_types"].add("forever")
    elif "keep " in lower and not slots["loop_counts"]:
        slots["loop_types"].add("forever")

    if any(phrase in lower for phrase in ACTION_LEXICON["show"]) or re.search(r"\bshow\b", lower):
        if not is_broadcast_phrase or "show up" in lower or "show yourself" in lower:
            slots["action_types"].add("show")
    if any(phrase in lower for phrase in ACTION_LEXICON["hide"]) or re.search(r"\bhide\b", lower):
        slots["action_types"].add("hide")
    if has_any_phrase(lower, ACTION_LEXICON["glide"]):
        slots["action_types"].add("glide")
    if has_any_phrase(lower, ACTION_LEXICON["go_to"]):
        slots["action_types"].add("go_to")
    if "position where x is" in lower and "y is" in lower:
        slots["action_types"].add("go_to")
    if "move to the middle" in lower or "middle of the screen" in lower:
        slots["action_types"].add("go_to")
        slots["numeric_values"].add("0")
    if has_any_phrase(lower, ACTION_LEXICON["move"]) and ("step" in lower or "backward" in lower or "forward" in lower):
        slots["action_types"].add("move")
    if "move " in lower and "steps" in lower:
        slots["action_types"].add("move")
    if has_any_phrase(lower, ACTION_LEXICON["turn_left"]):
        slots["action_types"].add("turn_left")
    if has_any_phrase(lower, ACTION_LEXICON["turn_right"]):
        slots["action_types"].add("turn_right")
    if "point in direction" in lower:
        slots["action_types"].add("point_direction")
    if has_any_phrase(lower, ACTION_LEXICON["point_towards"]):
        slots["action_types"].add("point_towards")
    if "point left" in lower or "point to the left" in lower:
        slots["action_types"].add("point_direction")
    if "point right" in lower or "point to the right" in lower:
        slots["action_types"].add("point_direction")
    if "face to the right" in lower or "facing to the right" in lower:
        slots["action_types"].add("point_direction")
    if "face to the left" in lower or "facing to the left" in lower:
        slots["action_types"].add("point_direction")
    if "face straight" in lower or "point straight" in lower:
        slots["action_types"].add("point_direction")
        slots["numeric_values"].add("0")
    if has_any_phrase(lower, ACTION_LEXICON["say"]):
        slots["action_types"].add("say")
    if has_any_phrase(lower, ACTION_LEXICON["think"]):
        slots["action_types"].add("think")
    if has_any_phrase(lower, ACTION_LEXICON["wait"]):
        slots["action_types"].add("wait")
    if has_any_phrase(lower, ACTION_LEXICON["switch_costume"]) or costume_targets:
        slots["action_types"].add("switch_costume")
        slots["costume_names"].update(costume_targets)
    if has_any_phrase(lower, ACTION_LEXICON["switch_backdrop"]):
        slots["action_types"].add("switch_backdrop")
        m = re.search(r"(?:change the backdrop to|switch backdrop to) ([a-z0-9_\-]+)", lower)
        if m:
            slots["backdrop_names"].add(normalize_name(m.group(1)))
    if has_any_phrase(lower, ACTION_LEXICON["next_costume"]):
        slots["action_types"].add("next_costume")
    if has_any_phrase(lower, ACTION_LEXICON["next_backdrop"]):
        slots["action_types"].add("next_backdrop")
    if (
        re.search(rf"\bset ({UNICODE_WORD_RE}) to\b", lower)
        or re.search(rf"\bmake ({UNICODE_WORD_RE}) zero\b", lower)
        or re.search(r"\bmake [^,.]+? zero\b", lower)
        or re.search(r"\bupdate [^,.]+ to\b", lower)
        or re.search(r"\bset [^,.=]+? = [^,.]+? to\b", lower)
        or re.search(r"\bchange [^,.]+ to\b", lower)
        or re.search(r"\bmake [^,.]+ equal to\b", lower)
        or re.search(r"\bmake [^,.]+ be\b", lower)
    ) and not costume_targets:
        slots["action_types"].add("set_variable")
    if (
        re.search(rf"\bchange ({UNICODE_WORD_RE}) by\b", lower)
        or "subtract" in lower
        or "add " in lower
        or has_any_phrase(lower, ACTION_LEXICON["change_variable"])
        or re.search(r"\b(?:make|set) the [^,.]+ go (?:up|down|left|right) by\b", lower)
    ) and not is_costume_or_backdrop_change and not re.search(r"\bchange [^,.]+ to\b", lower):
        slots["action_types"].add("change_variable")
    if re.search(r"\bmove (?:down|up|left|right)\b", lower):
        slots["action_types"].add("move")
        slots["action_types"].add("change_variable")
    if re.search(r"\bgo (?:down|up|left|right)\s+(-?\d+(?:\.\d+)?)\s+steps?\b", lower):
        slots["action_types"].add("move")
        slots["action_types"].add("change_variable")
        if "go down" in lower or "go up" in lower:
            slots["variable_names"].add("y")
        if "go left" in lower or "go right" in lower:
            slots["variable_names"].add("x")
    if has_any_phrase(lower, ACTION_LEXICON["play_sound"]):
        slots["action_types"].add("play_sound")
    if has_any_phrase(lower, ACTION_LEXICON["stop_sounds"]):
        slots["action_types"].add("stop_sounds")
    if has_any_phrase(lower, ACTION_LEXICON["broadcast"]):
        slots["action_types"].add("broadcast")
    if has_any_phrase(lower, ACTION_LEXICON["ask"]):
        slots["action_types"].add("ask")

    for m in re.finditer(rf"\bset ({UNICODE_WORD_RE}) to\b", lower):
        slots["variable_names"].add(normalize_name(m.group(1)))
    if not costume_targets:
        for m in re.finditer(r"\bchange ([^,.]+?) to\b", lower):
            slots["variable_names"].add(normalize_name(m.group(1)))
    for m in re.finditer(rf"\bchange ({UNICODE_WORD_RE}) by\b", lower):
        slots["variable_names"].add(normalize_name(m.group(1)))
    for m in re.finditer(r"\b(?:decrease|reduce|lower) ([^,.]+?) by\b", lower):
        slots["variable_names"].add(normalize_name(m.group(1)))
    for m in re.finditer(r"\btake away [^,.]* from ([^,.]+?)(?:,|\.| then| and|$)", lower):
        slots["variable_names"].add(normalize_name(m.group(1)))
    for m in re.finditer(r"\bupdate ([^,.]+?) to\b", lower):
        slots["variable_names"].add(normalize_name(m.group(1)))
    for m in re.finditer(rf"\bmake ({UNICODE_WORD_RE}) zero\b", lower):
        slots["variable_names"].add(normalize_name(m.group(1)))
    for m in re.finditer(r"\bmake ([^,.]+?) zero\b", lower):
        slots["variable_names"].add(normalize_name(m.group(1)))
    for m in re.finditer(rf"\bsubtract [^,.]* from ({UNICODE_WORD_RE})(?:,|\.| then| and|$)", lower):
        slots["variable_names"].add(normalize_name(m.group(1)))
    for m in re.finditer(r"\bset(?:ting)? the ([^,.]+?) to\b", lower):
        slots["variable_names"].add(normalize_name(m.group(1)))
    for m in re.finditer(r"\bset ([^,.=]+? = [^,.]+?) to\b", lower):
        slots["variable_names"].add(normalize_name(m.group(1)))
    for m in re.finditer(r"\bmake ([^,.]+?) equal to\b", lower):
        slots["variable_names"].add(normalize_name(m.group(1)))
    for m in re.finditer(r"\bmake ([^,.]+?) be\b", lower):
        slots["variable_names"].add(normalize_name(m.group(1)))
    for m in re.finditer(r"\b(?:make|set) the ([^,.]+?) go (?:up|down|left|right) by\b", lower):
        slots["variable_names"].add(normalize_name(m.group(1)))
    for m in re.finditer(r"\badd [^,.]* to ([^,.]+?)(?:,|\.| then| and|$)", lower):
        slots["variable_names"].add(normalize_name(m.group(1)))
    if re.search(r"\bx\s*(?:=|is)\s*-?\d", lower) and re.search(r"\by\s*(?:=|is)\s*-?\d", lower):
        slots["action_types"].add("go_to")
        slots["numeric_values"].add("0") if "middle of the screen" in lower else None
    for builtin in ["x", "y", "size", "volume"]:
        if re.search(rf"\b(?:set|change|make)\s+{builtin}\b", lower) or re.search(rf"\bfrom {builtin}(?:,|\.| then| and|$)", lower):
            slots["variable_names"].add(builtin)
    for variable_name, phrases in VARIABLE_DIRECTION_MAP.items():
        if has_any_phrase(lower, phrases):
            slots["variable_names"].add(variable_name)

    if has_any_phrase(lower, CONDITION_LEXICON["touching"]):
        slots["condition_types"].add("touching")
    if has_any_phrase(lower, CONDITION_LEXICON["edge"]):
        slots["condition_types"].add("edge")
    if ("if" in lower or "until" in lower) and "key" in lower and has_any_phrase(
        lower, CONDITION_LEXICON["key_pressed"]
    ):
        slots["condition_types"].add("key_pressed")
    if "mouse" in lower and ("until" in lower or "wait until" in lower):
        slots["condition_types"].add("mouse_down")
    if has_any_phrase(lower, CONDITION_LEXICON["mouse_down"]):
        slots["condition_types"].add("mouse_down")
    if ("if" in lower or "until" in lower) and has_any_phrase(lower, CONDITION_LEXICON["random"]):
        slots["condition_types"].add("random")
    if ("if" in lower or "until" in lower) and has_any_phrase(lower, CONDITION_LEXICON["answer"]):
        slots["condition_types"].add("answer")
    if " if " in f" {lower} " or lower.startswith("if ") or "until" in lower:
        if "equals" in lower or (" is " in lower and " key is pressed" not in lower and " key is not pressed" not in lower) or " is not " in lower:
            slots["condition_types"].add("equals")
        if has_any_phrase(lower, CONDITION_LEXICON["greater_than"]):
            slots["condition_types"].add("greater_than")
        if has_any_phrase(lower, CONDITION_LEXICON["less_than"]):
            slots["condition_types"].add("less_than")
    if "costume number" in lower:
        slots["condition_types"].add("equals")

    for m in re.finditer(r'(?:broadcast|send (?:out )?the message|send out) "?([^",.]+?)"?(?:,|\.| then| and|$)', lower):
        slots["broadcast_names"].add(normalize_name(m.group(1)))

    slots["variable_names"] = {name for name in slots["variable_names"] if is_reasonable_variable_name(name)}
    slots["costume_names"] = {normalize_name(name) for name in slots["costume_names"] if normalize_name(name)}
    slots["backdrop_names"] = {normalize_name(name) for name in slots["backdrop_names"] if normalize_name(name)}
    slots["broadcast_names"] = {normalize_name(name) for name in slots["broadcast_names"] if normalize_name(name)}

    manual_numeric_values = set(slots["numeric_values"])
    identifier_names = set()
    if slots["event_target"]:
        identifier_names.add(slots["event_target"])
    for name_slot in ["variable_names", "costume_names", "backdrop_names", "broadcast_names"]:
        identifier_names.update(slots[name_slot])
    identifier_names.update(extract_sound_targets(lower))
    masked_lower = mask_identifier_numbers(lower, identifier_names)
    slots["numeric_values"] = extract_numbers(masked_lower) | manual_numeric_values
    directional_values = add_signed_directional_numbers(set(), lower)
    slots["numeric_values"].update(directional_values)
    for signed in directional_values:
        if signed.startswith("-"):
            slots["numeric_values"].discard(signed[1:])

    return slots


def set_similarity(left: set[str], right: set[str]) -> float | None:
    if not left and not right:
        return None
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def scalar_similarity(left: str | None, right: str | None) -> float | None:
    if not left and not right:
        return None
    return 1.0 if left == right else 0.0


def guided_name_similarity(pseudo_values: set[str], nl_values: set[str], nl_text: str) -> float | None:
    if not pseudo_values and not nl_values:
        return None
    if not pseudo_values:
        return 0.0
    normalized_text = normalize_name(nl_text)
    matched = 0
    for value in pseudo_values:
        norm_value = normalize_name(value)
        if norm_value in nl_values or norm_value in normalized_text:
            matched += 1
    return matched / len(pseudo_values)


def compute_slot_scores(pseudo_slots: dict[str, Any], nl_slots: dict[str, Any], nl_text: str) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    scores["event_type"] = scalar_similarity(pseudo_slots["event_type"], nl_slots["event_type"])
    scores["event_target"] = scalar_similarity(pseudo_slots["event_target"], nl_slots["event_target"])
    guided_slots = {"variable_names", "costume_names", "backdrop_names", "broadcast_names"}
    for slot in SLOT_ORDER[2:]:
        if slot in {"loop_types", "loop_counts"}:
            scores[slot] = set_similarity(set(pseudo_slots[slot]), set(nl_slots[slot]))
            continue
        if slot in guided_slots:
            scores[slot] = guided_name_similarity(set(pseudo_slots[slot]), set(nl_slots[slot]), nl_text)
        else:
            scores[slot] = set_similarity(set(pseudo_slots[slot]), set(nl_slots[slot]))
    return scores


def serialize_slots(slots: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in slots.items():
        if isinstance(value, set):
            out[key] = sorted(value)
        else:
            out[key] = value
    return out


def merge_annotations(rows: list[dict[str, Any]], annotations: list[dict[str, Any]], fields: list[str]) -> None:
    by_key = {row["key"]: row for row in annotations}
    for row in rows:
        extra = by_key.get(row["key"])
        if not extra:
            continue
        for field in fields:
            if field in extra:
                row[field] = extra[field]


def summarize_results(results: list[dict[str, Any]], summary_path: Path, group_by_field: str | None) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["overall"] = results
    if group_by_field:
        for row in results:
            groups[str(row.get(group_by_field, "missing"))].append(row)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "metric", "value"])
        for group_name, rows in groups.items():
            if not rows:
                continue
            writer.writerow([group_name, "num_examples", len(rows)])
            writer.writerow(
                [group_name, "avg_alignment_score", sum(r["alignment_score"] for r in rows) / len(rows)]
            )
            writer.writerow(
                [group_name, "exact_alignment_rate", sum(r["exact_alignment"] for r in rows) / len(rows)]
            )
            writer.writerow(
                [group_name, "high_confidence_rate", sum(r["high_confidence_alignment"] for r in rows) / len(rows)]
            )
            for slot in SLOT_ORDER:
                values = [r["slot_scores"][slot] for r in rows if r["slot_scores"][slot] is not None]
                if values:
                    writer.writerow([group_name, f"{slot}_avg_score", sum(values) / len(values)])
                    writer.writerow([group_name, f"{slot}_coverage", len(values) / len(rows)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Rule-based semantic alignment validation for NL-pseudocode pairs.")
    ap.add_argument("--input", type=Path, required=True, help="Input JSONL with key, nl, pseudocode.")
    ap.add_argument("--output", type=Path, required=True, help="Per-example semantic alignment JSONL.")
    ap.add_argument("--summary", type=Path, required=True, help="Summary CSV output.")
    ap.add_argument("--mismatches", type=Path, required=True, help="Lowest-scoring examples JSONL.")
    ap.add_argument("--annotations", type=Path, help="Optional JSONL keyed by key for extra fields like nl_quality_bucket.")
    ap.add_argument(
        "--annotation-fields",
        nargs="+",
        default=["nl_quality_bucket", "nl_quality_score", "flesch", "fk_grade"],
        help="Fields to merge from the annotation JSONL.",
    )
    ap.add_argument("--group-by-field", default=None, help="Optional field to group summary rows by.")
    ap.add_argument("--mismatch-limit", type=int, default=250, help="How many lowest-scoring examples to save.")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of examples.")
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.annotations:
        merge_annotations(rows, load_jsonl(args.annotations), args.annotation_fields)

    results: list[dict[str, Any]] = []
    for row in rows:
        pseudo_slots = extract_pseudocode_slots(row["pseudocode"])
        nl_slots = extract_nl_slots(row["nl"])
        slot_scores = compute_slot_scores(pseudo_slots, nl_slots, row["nl"])
        comparable = [value for value in slot_scores.values() if value is not None]
        alignment_score = sum(comparable) / len(comparable) if comparable else 1.0
        exact_alignment = all(math.isclose(value, 1.0) for value in comparable) if comparable else True
        mismatch_slots = [slot for slot, value in slot_scores.items() if value is not None and value < 1.0]

        out = dict(row)
        out["pseudo_slots"] = serialize_slots(pseudo_slots)
        out["nl_slots"] = serialize_slots(nl_slots)
        out["slot_scores"] = slot_scores
        out["alignment_score"] = alignment_score
        out["exact_alignment"] = exact_alignment
        out["high_confidence_alignment"] = alignment_score >= 0.85
        out["mismatch_slots"] = mismatch_slots
        results.append(out)

    write_jsonl(results, args.output)
    summarize_results(results, args.summary, args.group_by_field)
    mismatch_rows = sorted(results, key=lambda row: (row["alignment_score"], row["key"]))[: args.mismatch_limit]
    write_jsonl(mismatch_rows, args.mismatches)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "summary": str(args.summary),
                "mismatches": str(args.mismatches),
                "num_examples": len(results),
                "avg_alignment_score": sum(r["alignment_score"] for r in results) / max(1, len(results)),
                "exact_alignment_rate": sum(r["exact_alignment"] for r in results) / max(1, len(results)),
            }
        )
    )


if __name__ == "__main__":
    main()