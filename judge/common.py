from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_json_to_dict(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    s = re.sub(r'<think>.*?</think>\s*', '', s, flags=re.DOTALL)
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


def build_conversation_text(instruction: str, response: str) -> str:
    return f"user: {instruction}\n\nassistant: {response}"


def create_pointwise_judge_prompt(conversation: str, rubric_items: str) -> str:
    return f"""
Your job is to look at a conversation and a set of rubric items, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
{conversation}

# Rubric items
{rubric_items}

# Instructions
Return a json object. For each rubric item i (starting from 1), keys must be exactly "explanation_i" and "criteria_met_i" for each i and it includes two top-level fields in the JSON object:
- The "explanation_i" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met_i" field should be a boolean indicating (true/false) whether the response meets the criteria of the rubric item.

# Final Output Format
{{
  "explanation_1": "...",
  "criteria_met_1": true/false,
  "explanation_2": "...",
  "criteria_met_2": true/false,
  ...
}}

Return just the json object. Do not include any other text.
""".strip()


def count_rubric_items(rubric_text: str) -> int:
    return len(re.findall(r"^\s*\d+\.\s", rubric_text, flags=re.MULTILINE))


def compute_pointwise_reward(parsed: Dict[str, Any], rubric_count: int) -> Tuple[float, str]:
    if not parsed or rubric_count <= 0:
        return 0.0, "empty/invalid"
    
    details = []
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


def format_is_valid(parsed: Dict[str, Any], rubric_count: int) -> bool:
    if rubric_count <= 0 or not parsed:
        return False
    
    for i in range(1, rubric_count + 1):
        if f"criteria_met_{i}" not in parsed or f"explanation_{i}" not in parsed:
            return False
        if not isinstance(parsed.get(f"criteria_met_{i}"), bool):
            return False
    
    return True
