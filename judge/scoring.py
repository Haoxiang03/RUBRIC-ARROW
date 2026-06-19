from __future__ import annotations

import json
import math
import re
from typing import Any


RUBRIC_ITEM = re.compile(r"(?m)^\s*(\d+)\.\s+(.+?)\s*$")
RUBRIC_TAG = re.compile(r"\[([^\]]+)\]\s*$")
CRITERIA_KEY = re.compile(r'"criteria_met_(\d+)"\s*:')
THINK_BLOCK = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)

TAG_WEIGHTS = {
    "hard rule": 3.0,
    "principle": 1.0,
}


def rubric_weights(rubric_text: str) -> list[float]:
    items = RUBRIC_ITEM.findall(rubric_text)
    if not items:
        raise ValueError("Rubric must contain at least one numbered item.")
    indices = [int(index) for index, _ in items]
    if indices != list(range(1, len(items) + 1)):
        raise ValueError("Rubric items must be numbered consecutively from 1.")

    weights = []
    for _, text in items:
        match = RUBRIC_TAG.search(text)
        if not match:
            raise ValueError("Every rubric item must end with [Hard Rule] or [Principle].")
        tag = match.group(1).strip().lower()
        if tag not in TAG_WEIGHTS:
            raise ValueError("Rubric tags must be Hard Rule or Principle.")
        weights.append(TAG_WEIGHTS[tag])
    return weights


def judge_prompt(conversation: str, rubric_text: str) -> str:
    return f"""Your job is to score the final assistant response in the conversation against each rubric item.

# Conversation
{conversation}

# Rubric items
{rubric_text}

# Instructions
Return one JSON object. For each rubric item i, include exactly:
- "explanation_i": a brief justification.
- "criteria_met_i": a boolean.

Return only the JSON object."""


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
    entries = logprobs["content"]
    probabilities: dict[int, tuple[float, float]] = {}
    context = ""
    active_index: int | None = None

    for entry in entries:
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
    rubric_text: str,
) -> tuple[float, bool]:
    weights = rubric_weights(rubric_text)
    try:
        parse_judgement(text, len(weights))
        probabilities = boolean_probabilities(logprobs)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 0.0, False

    if set(probabilities) != set(range(1, len(weights) + 1)):
        return 0.0, False

    score = sum(
        weight * (probabilities[index][0] - probabilities[index][1])
        for index, weight in enumerate(weights, start=1)
    )
    return score, True
