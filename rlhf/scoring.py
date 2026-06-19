from __future__ import annotations

import json
import math
import re
from typing import Any


ITEM_START = re.compile(r"^(\d+)\.\s+(.+)$")
CATEGORY_TAG = re.compile(r"\[(Hard Rule|Principle)\]\s*$", re.IGNORECASE)
CRITERIA_KEY = re.compile(r'"criteria_met_(\d+)"\s*:')
THINK_BLOCK = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)

TAG_WEIGHTS = {
    "hard rule": 3.0,
    "principle": 1.0,
}


def parse_rubrics(block: str) -> list[str]:
    items: list[tuple[int, list[str]]] = []
    current_index: int | None = None
    current_parts: list[str] = []

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = ITEM_START.match(line)
        if match:
            if current_index is not None:
                items.append((current_index, current_parts))
            current_index = int(match.group(1))
            current_parts = [match.group(2)]
        elif current_index is not None:
            current_parts.append(line)
        else:
            raise ValueError("Rubric text must start with a numbered item.")

    if current_index is not None:
        items.append((current_index, current_parts))
    if not items:
        raise ValueError("Rubric text is empty.")

    parsed = []
    for expected, (index, parts) in enumerate(items, start=1):
        if index != expected:
            raise ValueError("Rubric items must be numbered consecutively from 1.")
        parsed.append(f"{index}. {' '.join(parts)}")
    return parsed


def rubric_weights(items: list[str]) -> list[float]:
    weights = []
    for item in items:
        match = CATEGORY_TAG.search(item)
        if not match:
            raise ValueError("Rubric tags must be Hard Rule or Principle.")
        weights.append(TAG_WEIGHTS[match.group(1).lower()])
    return weights


def judge_prompt(prompt: str, response: str, rubric: str) -> str:
    return f"""Score the final assistant response against every rubric item.

# Conversation
user: {prompt}

assistant: {response}

# Rubric items
{rubric}

# Instructions
For each item i, return exactly:
- "explanation_i": a brief justification.
- "criteria_met_i": true only when every requirement in the item is met.

Examples introduced by phrases such as "for example", "such as", or "including" are not exhaustive.
Return only one JSON object."""


def parse_judgement(text: str, rubric_count: int) -> dict[str, Any]:
    parsed = json.loads(THINK_BLOCK.sub("", text).strip())
    if not isinstance(parsed, dict):
        raise ValueError("Judge output must be a JSON object.")

    expected = {
        key
        for index in range(1, rubric_count + 1)
        for key in (f"explanation_{index}", f"criteria_met_{index}")
    }
    if set(parsed) != expected:
        raise ValueError("Judge output keys do not match the rubric.")

    for index in range(1, rubric_count + 1):
        if not isinstance(parsed[f"explanation_{index}"], str):
            raise ValueError("Judge explanations must be strings.")
        if not isinstance(parsed[f"criteria_met_{index}"], bool):
            raise ValueError("Judge decisions must be booleans.")
    return parsed


def boolean_probabilities(logprobs: dict[str, Any]) -> dict[int, tuple[float, float]]:
    probabilities: dict[int, tuple[float, float]] = {}
    context = ""
    active_index: int | None = None

    for entry in logprobs["content"]:
        token = entry["token"]
        context = (context + token)[-256:]

        if active_index is None:
            matches = CRITERIA_KEY.findall(context)
            if matches:
                index = int(matches[-1])
                if index not in probabilities:
                    active_index = index
            continue

        candidates = {
            candidate["token"].strip(): float(candidate["logprob"])
            for candidate in entry["top_logprobs"]
        }
        candidates.setdefault(token.strip(), float(entry["logprob"]))
        if "true" not in candidates or "false" not in candidates:
            continue

        probabilities[active_index] = (
            math.exp(candidates["true"]),
            math.exp(candidates["false"]),
        )
        active_index = None

    return probabilities


def score_judgement(
    text: str,
    logprobs: dict[str, Any],
    weights: list[float],
) -> tuple[float, bool]:
    try:
        parse_judgement(text, len(weights))
        probabilities = boolean_probabilities(logprobs)
    except (KeyError, TypeError, ValueError):
        return 0.0, False

    if set(probabilities) != set(range(1, len(weights) + 1)):
        return 0.0, False

    weight_sum = sum(weights)
    score = sum(
        weight * (probabilities[index][0] - probabilities[index][1])
        for index, weight in enumerate(weights, start=1)
    )
    return score / weight_sum, True
