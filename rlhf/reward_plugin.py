"""Reward plugin for GRPO answer training with rubric-based judge scoring."""

import os
from typing import Any, Dict, List, Sequence

import requests

from swift.plugin import ORM, orms


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except Exception:
        return [value]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


class RubricAnswerReward(ORM):
    def __init__(self) -> None:
        super().__init__()

        host = os.environ.get("REWARD_SERVER_HOST", "127.0.0.1")
        port = os.environ.get("REWARD_SERVER_PORT", "8000")
        self.endpoint = f"http://{host}:{port}/score"
        self.timeout = float(os.environ.get("REWARD_TIMEOUT", "300"))
        self._session = requests.Session()
        self._call_count = 0
        self._global_rank = _as_int(os.environ.get("RANK"), 0)
        self._local_rank = _as_int(os.environ.get("LOCAL_RANK"), 0)

        print(f"[RubricAnswerReward] endpoint: {self.endpoint}", flush=True)

    def _extract_user_prompt(self, msg_list: Any) -> str:
        if isinstance(msg_list, list):
            for msg in msg_list:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return _as_text(msg.get("content"))
        return ""

    def _pick_value(
        self,
        seq: Sequence[Any],
        first_idx: int,
        group_idx: int,
        default: str = "",
    ) -> str:
        if not seq:
            return default
        if 0 <= first_idx < len(seq):
            return _as_text(seq[first_idx])
        if 0 <= group_idx < len(seq):
            return _as_text(seq[group_idx])
        return _as_text(seq[-1])

    def _pick_raw(
        self,
        seq: Sequence[Any],
        first_idx: int,
        group_idx: int,
        default: Any = None,
    ) -> Any:
        if not seq:
            return default
        if 0 <= first_idx < len(seq):
            return seq[first_idx]
        if 0 <= group_idx < len(seq):
            return seq[group_idx]
        return seq[-1]

    def _normalize_rubrics(self, raw: Any) -> List[str]:
        if isinstance(raw, str):
            text = raw.strip()
            return [text] if text else []
        if isinstance(raw, list):
            out: List[str] = []
            for item in raw:
                if isinstance(item, str):
                    txt = item.strip()
                    if txt:
                        out.append(txt)
                elif item is not None:
                    txt = str(item).strip()
                    if txt:
                        out.append(txt)
            return out
        if raw is None:
            return []
        txt = str(raw).strip()
        return [txt] if txt else []

    def _request_rewards(
        self,
        prompt: str,
        rubrics: List[str],
        completions: List[str],
    ) -> List[float]:
        if not prompt or not rubrics:
            return [0.0] * len(completions)

        try:
            resp = self._session.post(
                self.endpoint,
                json={
                    "prompt": prompt,
                    "rubrics": rubrics,
                    "completions": completions,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            rewards = resp.json().get("rewards", [])
            rewards = [float(x) for x in rewards]
            if len(rewards) != len(completions):
                rewards = (rewards + [0.0] * len(completions))[: len(completions)]
            return rewards
        except Exception as exc:
            print(f"[RubricAnswerReward] request error: {exc}", flush=True)
            return [0.0] * len(completions)

    def __call__(
        self,
        completions: Sequence[str],
        prompt: Sequence[str] | None = None,
        rubrics: Sequence[Any] | None = None,
        original_prompt: Sequence[str] | None = None,
        **kwargs,
    ) -> List[float]:
        self._call_count += 1

        completions_list = [_as_text(x) for x in (completions or [])]
        n_completions = len(completions_list)
        if n_completions == 0:
            return []

        rubrics_list = _to_list(rubrics or kwargs.get("rubrics") or [])
        original_prompt_list = _to_list(
            original_prompt
            or kwargs.get("original_prompt")
            or prompt
            or kwargs.get("prompt")
            or []
        )
        messages = _to_list(kwargs.get("messages") or [])
        prompt_ids = _to_list(kwargs.get("prompt_id") or kwargs.get("prompt_ids") or [])

        all_rewards: List[float] = [0.0] * n_completions

        print(
            f"[RubricAnswerReward][rank={self._global_rank},local_rank={self._local_rank}] "
            f"call={self._call_count} n_completions={n_completions}",
            flush=True,
        )

        if prompt_ids and len(prompt_ids) == n_completions:
            pid_to_indices: Dict[str, List[int]] = {}
            for idx, pid in enumerate(prompt_ids):
                pid_to_indices.setdefault(str(pid), []).append(idx)

            for group_idx, (_, idxs) in enumerate(pid_to_indices.items()):
                first = idxs[0]
                group_completions = [completions_list[i] for i in idxs]

                prompt_text = self._pick_value(original_prompt_list, first, group_idx)
                if not prompt_text:
                    prompt_text = self._extract_user_prompt(
                        self._pick_raw(messages, first, group_idx, default=[])
                    )

                rubrics_raw = self._pick_raw(rubrics_list, first, group_idx, default=[])
                rubrics_text = self._normalize_rubrics(rubrics_raw)

                group_rewards = self._request_rewards(
                    prompt=prompt_text,
                    rubrics=rubrics_text,
                    completions=group_completions,
                )
                for i, rwd in zip(idxs, group_rewards):
                    all_rewards[i] = rwd
        else:
            n_samples = 0
            if isinstance(messages, list) and messages:
                n_samples = len(messages)
            elif rubrics_list:
                n_samples = len(rubrics_list)
            elif original_prompt_list:
                n_samples = len(original_prompt_list)

            if n_samples <= 0:
                return all_rewards

            k = max(1, n_completions // n_samples)
            for sample_idx in range(n_samples):
                start = sample_idx * k
                end = (sample_idx + 1) * k if sample_idx < n_samples - 1 else n_completions
                if start >= n_completions:
                    break
                end = min(end, n_completions)
                if end <= start:
                    continue

                group_completions = completions_list[start:end]

                prompt_text = self._pick_value(original_prompt_list, sample_idx, sample_idx)
                if not prompt_text and sample_idx < len(messages):
                    prompt_text = self._extract_user_prompt(messages[sample_idx])

                rubrics_raw = self._pick_raw(rubrics_list, sample_idx, sample_idx, default=[])
                rubrics_text = self._normalize_rubrics(rubrics_raw)

                group_rewards = self._request_rewards(
                    prompt=prompt_text,
                    rubrics=rubrics_text,
                    completions=group_completions,
                )
                for offset, rwd in enumerate(group_rewards):
                    all_rewards[start + offset] = rwd

        if len(all_rewards) != n_completions:
            all_rewards = (all_rewards + [0.0] * n_completions)[:n_completions]

        print(
            f"[RubricAnswerReward][rank={self._global_rank},local_rank={self._local_rank}] "
            f"call={self._call_count} rewards={all_rewards}",
            flush=True,
        )

        return all_rewards


orms["rubric_answer_reward"] = RubricAnswerReward
