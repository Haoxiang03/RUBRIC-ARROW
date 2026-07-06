"""
Reward plugin for GRPO rubric generation.
"""

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


class RubricPairwiseReward(ORM):
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
        self._is_main_process = self._global_rank == 0 and self._local_rank == 0

        print(f"[RubricPairwiseReward] endpoint: {self.endpoint}", flush=True)

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

    def _request_rewards(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        completions: List[str],
    ) -> List[float]:
        if not prompt or not chosen or not rejected:
            return [-1.0] * len(completions)

        try:
            resp = self._session.post(
                self.endpoint,
                json={
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "completions": completions,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            rewards = resp.json().get("rewards", [])
            rewards = [float(x) for x in rewards]
            if len(rewards) != len(completions):
                rewards = (rewards + [-1.0] * len(completions))[: len(completions)]
            return rewards
        except Exception as exc:
            print(f"[RubricPairwiseReward] request error: {exc}", flush=True)
            return [-1.0] * len(completions)

    def __call__(
        self,
        completions: Sequence[str],
        prompt: Sequence[str] | None = None,
        chosen: Sequence[str] | None = None,
        rejected: Sequence[str] | None = None,
        original_prompt: Sequence[str] | None = None,
        **kwargs,
    ) -> List[float]:
        self._call_count += 1

        completions_list = [_as_text(x) for x in (completions or [])]
        n_completions = len(completions_list)
        if n_completions == 0:
            return []

        chosen_list = _to_list(chosen or kwargs.get("chosen") or [])
        rejected_list = _to_list(rejected or kwargs.get("rejected") or [])
        original_prompt_list = _to_list(
            original_prompt
            or kwargs.get("original_prompt")
            or prompt
            or kwargs.get("prompt")
            or []
        )
        messages = _to_list(kwargs.get("messages") or [])
        prompt_ids = _to_list(kwargs.get("prompt_id") or kwargs.get("prompt_ids") or [])

        all_rewards: List[float] = [-1.0] * n_completions

        print(
            f"[RubricPairwiseReward] call={self._call_count} n_completions={n_completions}",
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

                chosen_text = self._pick_value(chosen_list, first, group_idx)
                rejected_text = self._pick_value(rejected_list, first, group_idx)

                group_rewards = self._request_rewards(
                    prompt=prompt_text,
                    chosen=chosen_text,
                    rejected=rejected_text,
                    completions=group_completions,
                )
                for i, rwd in zip(idxs, group_rewards):
                    all_rewards[i] = rwd
        else:
            n_samples = 0
            if isinstance(messages, list) and messages:
                n_samples = len(messages)
            elif chosen_list:
                n_samples = len(chosen_list)
            elif rejected_list:
                n_samples = len(rejected_list)
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
                chosen_text = self._pick_value(chosen_list, sample_idx, sample_idx)
                rejected_text = self._pick_value(rejected_list, sample_idx, sample_idx)

                group_rewards = self._request_rewards(
                    prompt=prompt_text,
                    chosen=chosen_text,
                    rejected=rejected_text,
                    completions=group_completions,
                )
                for offset, rwd in enumerate(group_rewards):
                    all_rewards[start + offset] = rwd

        if len(all_rewards) != n_completions:
            all_rewards = (all_rewards + [-1.0] * n_completions)[:n_completions]

        # Print rewards on every rank so the full distributed batch is visible in logs.
        # Each rank logs its local reward list; concatenate logs across ranks to get all rewards.
        valid_rewards = [float(x) for x in all_rewards]
        print(
            f"[RubricPairwiseReward][rank={self._global_rank},local_rank={self._local_rank}] "
            f"call={self._call_count} rewards={valid_rewards}",
            flush=True,
        )

        return all_rewards


orms["rubric_pairwise_reward"] = RubricPairwiseReward
