from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence, Tuple


RUBRIC_TEMPLATE = (
    "Your task is to extract a set of rubric-style instructions from a user's request.\n"
    "These rubrics will be used as evaluation criteria to check if a response fully meets the request.\n"
    "Every rubric item must be a universal principle. If any rubric still contains topic-specific references (e.g., names, places, myths, numbers, historical facts), it is automatically invalid.\n"
    "\n"
    "- **Two Distinct Categories:**\n"
    "  - [Hard Rule]: Derived strictly from explicit requirements stated in the <request> (format, length, structure, forbidden/required elements, etc.).\n"
    "  - [Principle]: Derived by abstracting any concrete cues into domain-agnostic quality criteria (e.g., clarity, correctness, sound reasoning, pedagogy).\n"
    "\n"
    "- **Comprehensiveness:**\n"
    "  The rubric must cover all critical aspects implied by the request and examples, including explicit requirements and implicit quality standards.\n"
    "\n"
    "- **Conciseness & Uniqueness:**\n"
    "  Each rubric must capture a distinct evaluation criterion. Overlapping or redundant criteria must be merged into a single rubric. Wording must be precise and free of repetition.\n"
    "\n"
    "- **Format Requirements:**\n"
    "  - Use a numbered list.\n"
    "  - Each item starts with \"The response\" phrased in third person.\n"
    "  - Append [Hard Rule] or [Principle] at the end of each item.\n"
    "  - Do not include reasoning, explanations, or examples in the final output - only the rubrics.\n"
    "\n"
    "Here is the request:\n"
    "{prompt}\n"
    "\n"
    "Please generate the rubrics for the above request."
)


RUBRIC_START_RE = re.compile(r"^\s*(\d+)\.\s*(.*)$")
RUBRIC_TAG_RE = re.compile(r"\[(Hard Rule|Principle)\]\s*$", flags=re.IGNORECASE)
RUBRIC_HEAD_RE = re.compile(r"(?i)^the response\b")


def build_rubric_generation_prompt(prompt: str) -> str:
    return RUBRIC_TEMPLATE.format(prompt=(prompt or ""))


def _normalize_rubric_text(rubric_text: str) -> str:
    s = (rubric_text or "").strip()
    # Remove reasoning wrappers and keep only final answer content if present.
    s = re.sub(r"<think>.*?</think>\s*", "", s, flags=re.DOTALL)
    s = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", s, flags=re.IGNORECASE)
    answer_match = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", s, flags=re.IGNORECASE)
    if answer_match:
        s = answer_match.group(1).strip()
    return s


def build_conversation_text(instruction: str, response: str) -> str:
    return f"user: {instruction}\n\nassistant: {response}"


def create_pointwise_judge_prompt(conversation: str, rubric_items: str) -> str:
    return f"""
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


def parse_json_to_dict(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    s = re.sub(r"<think>.*?</think>\s*", "", s, flags=re.DOTALL)
    s = re.sub(r"^```json\s*|\s*```$", "", s)
    s = re.sub(r"^```[\w-]*\s*|\s*```$", "", s)
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


def _parse_numbered_items(rubric_text: str) -> List[Tuple[int, str]]:
    items: List[Tuple[int, str]] = []
    current_idx: int | None = None
    current_parts: List[str] = []

    for raw in (rubric_text or "").replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue

        m = RUBRIC_START_RE.match(line)
        if m:
            if current_idx is not None:
                items.append((current_idx, " ".join(current_parts).strip()))
            current_idx = int(m.group(1))
            current_parts = [m.group(2).strip()]
            continue

        if current_idx is None:
            return []
        current_parts.append(line)

    if current_idx is not None:
        items.append((current_idx, " ".join(current_parts).strip()))

    return items


def validate_rubric_format(rubric_text: str) -> Tuple[bool, List[str], str]:
    items = _parse_numbered_items(_normalize_rubric_text(rubric_text))
    if not items:
        return False, [], "missing_numbered_items"

    normalized_lines: List[str] = []
    for expected_idx, (idx, content) in enumerate(items, start=1):
        if idx != expected_idx:
            return False, [], f"non_consecutive_index_{idx}_expected_{expected_idx}"

        tag_match = RUBRIC_TAG_RE.search(content)
        if not tag_match:
            return False, [], f"missing_tag_item_{idx}"

        tag_raw = tag_match.group(1).strip().lower()
        tag = "Hard Rule" if tag_raw == "hard rule" else "Principle"

        body = content[: tag_match.start()].strip()
        if not RUBRIC_HEAD_RE.match(body):
            return False, [], f"item_{idx}_must_start_with_the_response"

        normalized_lines.append(f"{idx}. {body} [{tag}]")

    return True, normalized_lines, ""


def count_rubric_items(rubric_text: str) -> int:
    ok, items, _ = validate_rubric_format(rubric_text)
    if not ok:
        return 0
    return len(items)


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


def compute_pointwise_reward(parsed: Dict[str, Any], rubric_count: int) -> Tuple[float, str]:
    if not parsed or rubric_count <= 0:
        return 0.0, "empty/invalid"

    details: List[str] = []
    total = 0.0
    weight_sum = 0.0

    for i in range(1, rubric_count + 1):
        val = parsed.get(f"criteria_met_{i}", None)
        weight = 1.0

        if isinstance(val, bool):
            score_i = weight if val else -weight
            details.append(f"[{i}]{score_i:.0f}")
            total += score_i
            weight_sum += weight
        else:
            details.append(f"[{i}]?")
            weight_sum += weight

    final_score = total / weight_sum if weight_sum > 0 else 0.0
    detail_str = "+".join(details) + f"={final_score:.4f}"
    return final_score, detail_str
