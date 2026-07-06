from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence, Tuple

RUBRIC_START_RE = re.compile(r"^\s*(\d+)\.\s*(.*)$")


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _strip_wrappers(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"<think>.*?</think>\s*", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"^```(?:json|text|markdown)?\s*|\s*```$", "", s, flags=re.IGNORECASE)
    answer_match = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", s, flags=re.IGNORECASE)
    if answer_match:
        s = answer_match.group(1).strip()
    return s


def split_numbered_items(text: str) -> List[str]:
    lines = _strip_wrappers(text).replace("\r\n", "\n").split("\n")
    items: List[Tuple[int, List[str]]] = []
    current_idx: int | None = None
    current_parts: List[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m = RUBRIC_START_RE.match(line)
        if m:
            if current_idx is not None:
                items.append((current_idx, current_parts))
            current_idx = int(m.group(1))
            current_parts = [m.group(2).strip()]
            continue

        if current_idx is not None:
            current_parts.append(line)

    if current_idx is not None:
        items.append((current_idx, current_parts))

    if not items:
        return []

    normalized: List[str] = []
    for new_idx, (_, parts) in enumerate(items, start=1):
        body = " ".join(p.strip() for p in parts if p.strip())
        if body:
            normalized.append(f"{new_idx}. {body}")
    return normalized


def normalize_rubrics(raw_rubrics: Any) -> List[str]:
    blocks: List[str] = []
    if isinstance(raw_rubrics, str):
        blocks = [raw_rubrics]
    elif isinstance(raw_rubrics, list):
        for item in raw_rubrics:
            if isinstance(item, str):
                if item.strip():
                    blocks.append(item)
            elif isinstance(item, dict):
                for key in ("rubric", "text", "criterion"):
                    val = _as_text(item.get(key))
                    if val:
                        blocks.append(val)
                        break
    elif isinstance(raw_rubrics, dict):
        for key in ("rubric", "text", "criterion"):
            val = _as_text(raw_rubrics.get(key))
            if val:
                blocks.append(val)
                break

    all_items: List[str] = []
    for block in blocks:
        all_items.extend(split_numbered_items(block))

    # Re-index globally after concatenation to ensure sequential ids.
    reindexed: List[str] = []
    for idx, item in enumerate(all_items, start=1):
        m = RUBRIC_START_RE.match(item)
        body = m.group(2).strip() if m else item.strip()
        if body:
            reindexed.append(f"{idx}. {body}")
    return reindexed


def _weight_from_tags(tags: List[str]) -> float:
    tags = [str(t).strip().lower() for t in (tags or [])]
    if "hard rule" in tags:
        return 3.0
    if "principle" in tags:
        return 1.0
    return 1.0


def rubric_weights_from_items(rubric_items: Sequence[str]) -> List[float]:
    weights: List[float] = []
    for line in rubric_items:
        tags: List[str] = []
        for m in re.finditer(r"\[([^\]]+)\]", line or ""):
            tags.extend([t.strip() for t in (m.group(1) or "").split(",") if t.strip()])
        weights.append(_weight_from_tags(tags))
    return weights


def build_conversation_text(instruction: str, response: str) -> str:
    return f"user: {instruction}\n\nassistant: {response}"


GRADER_TEMPLATE = """
Your job is to look at a conversation and a set of rubric items, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object. For each rubric item i (starting from 1), keys must be exactly "explanation_i" and "criteria_met_i" for each i and it includes two top-level fields in the JSON object:
- The "explanation_i" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met_i" field should be a boolean indicating (true/false) whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria.

# Final Output Format (a single JSON object, not an array)
{
  "explanation_1": "...",
  "criteria_met_1": true/false,
  "explanation_2": "...",
  "criteria_met_2": true/false,
  ... repeat this pattern for every rubric item i in order (i = 1, 2, 3, ...)
}

# Final instruction
Return just the json object. Do not include any other text in the response.
""".strip()


def create_pointwise_judge_prompt(conversation: str, rubric_items: str) -> str:
    return (
        GRADER_TEMPLATE
        .replace("<<conversation>>", conversation)
        .replace("<<rubric_item>>", rubric_items)
    )


def parse_json_to_dict(s: str) -> Dict[str, Any]:
    s = _strip_wrappers(s)
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        s2 = s.replace("True", "true").replace("False", "false")
        try:
            return json.loads(s2)
        except Exception:
            return {}


def compute_pointwise_reward_from_json(
    parsed: Dict[str, Any],
    rubric_count: int,
    weights: Sequence[float],
) -> Tuple[float, str]:
    if not parsed or rubric_count <= 0:
        return 0.0, "empty/invalid"

    def _to_prob(v: Any) -> float | None:
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "yes"}:
                return 1.0
            if s in {"false", "no"}:
                return 0.0
            try:
                return max(0.0, min(1.0, float(s)))
            except Exception:
                return None
        return None

    total = 0.0
    total_weight = 0.0
    parts: List[str] = []
    for i in range(1, rubric_count + 1):
        w = float(weights[i - 1]) if i - 1 < len(weights) else 1.0
        total_weight += w

        tp = _to_prob(parsed.get(f"true_prob_{i}"))
        fp = _to_prob(parsed.get(f"false_prob_{i}"))

        if tp is None or fp is None:
            met = parsed.get(f"criteria_met_{i}")
            met_prob = _to_prob(met)
            if met_prob is not None:
                tp = met_prob
                fp = 1.0 - met_prob

        if tp is None or fp is None:
            parts.append(f"[{i}]?")
            continue

        step = (tp - fp) * w
        total += step
        parts.append(f"[{i}]tp={tp:.3f},fp={fp:.3f},w={w:.1f},s={step:+.3f}")
    if total_weight <= 0:
        return 0.0, "raw=0.0000 weight_sum=0.0000 norm=0.0000 (weight_sum<=0)"
    normalized = total / total_weight
    return normalized, (
        f"raw={total:.4f} weight_sum={total_weight:.4f} norm={normalized:.4f} | " + "+".join(parts)
    )
