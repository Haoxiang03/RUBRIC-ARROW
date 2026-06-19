from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Sequence

import requests

from swift.plugin import ORM, orms


class RubricPairwiseReward(ORM):
    def __init__(self) -> None:
        super().__init__()
        host = os.getenv("REWARD_SERVER_HOST", "127.0.0.1")
        port = os.getenv("REWARD_SERVER_PORT", "8000")
        self.endpoint = f"http://{host}:{port}/score"
        self.timeout = float(os.getenv("REWARD_TIMEOUT", "300"))
        self.session = requests.Session()

    def __call__(
        self,
        completions: Sequence[str],
        chosen: Sequence[str],
        rejected: Sequence[str],
        original_prompt: Sequence[str],
        prompt_id: Sequence[str],
        **_: Any,
    ) -> list[float]:
        size = len(completions)
        fields = {
            "chosen": chosen,
            "rejected": rejected,
            "original_prompt": original_prompt,
            "prompt_id": prompt_id,
        }
        for name, values in fields.items():
            if len(values) != size:
                raise ValueError(f"{name} must contain one value per completion.")

        groups: dict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(prompt_id):
            groups[str(value)].append(index)

        rewards = [-1.0] * size
        for indices in groups.values():
            first = indices[0]
            response = self.session.post(
                self.endpoint,
                json={
                    "prompt": original_prompt[first],
                    "chosen": chosen[first],
                    "rejected": rejected[first],
                    "completions": [completions[index] for index in indices],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            group_rewards = response.json()["rewards"]
            if len(group_rewards) != len(indices):
                raise ValueError("Reward server returned an unexpected number of rewards.")

            for index, reward in zip(indices, group_rewards):
                rewards[index] = float(reward)

        return rewards


orms["rubric_pairwise_reward"] = RubricPairwiseReward
