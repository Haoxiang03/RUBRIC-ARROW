from __future__ import annotations

import json
import math
import re
from typing import Any


RUBRIC_LINE = re.compile(
    r"^(\d+)\.\s+(The response\b.+?)\s+\[(Hard Rule|Principle)\]\s*$"
)
CRITERIA_KEY = re.compile(r'"criteria_met_(\d+)"\s*:')
THINK_BLOCK = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)
TAG_WEIGHTS = {"Hard Rule": 3.0, "Principle": 1.0}

MIN_AVERAGE_RUBRICS_FOR_BONUS = 5.0
SHORTEST_BONUS = 0.1


def parse_rubric(text: str) -> list[str]:
    normalized = THINK_BLOCK.sub("", text).strip()
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Rubric output is empty.")

    items = []
    for expected, line in enumerate(lines, start=1):
        match = RUBRIC_LINE.fullmatch(line)
        if not match or int(match.group(1)) != expected:
            raise ValueError("Rubrics must use consecutive numbered items in the required format.")
        items.append(f"{expected}. {match.group(2)} [{match.group(3)}]")
    return items


def rubric_weights(items: list[str]) -> list[float]:
    return [
        TAG_WEIGHTS["Hard Rule"] if item.endswith("[Hard Rule]") else TAG_WEIGHTS["Principle"]
        for item in items
    ]


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

    score = sum(
        weight * (probabilities[index][0] - probabilities[index][1])
        for index, weight in enumerate(weights, start=1)
    )
    return score, True


def assign_rewards(
    chosen_scores: list[float],
    rejected_scores: list[float],
    rubric_counts: list[int],
    valid: list[bool],
) -> list[float]:
    if not (
        len(chosen_scores)
        == len(rejected_scores)
        == len(rubric_counts)
        == len(valid)
    ):
        raise ValueError("Reward inputs must have equal lengths.")

    rewards = [-1.0] * len(valid)
    correct = [
        index
        for index in range(len(valid))
        if valid[index] and chosen_scores[index] > rejected_scores[index]
    ]
    for index in correct:
        rewards[index] = 1.0

    if correct:
        average_count = sum(rubric_counts[index] for index in correct) / len(correct)
        if average_count >= MIN_AVERAGE_RUBRICS_FOR_BONUS:
            shortest = min(rubric_counts[index] for index in correct)
            for index in correct:
                if rubric_counts[index] == shortest:
                    rewards[index] += SHORTEST_BONUS
    return rewards
