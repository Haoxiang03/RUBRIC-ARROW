from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Sequence

import requests

from swift.plugin import ORM, orms


def user_prompt(messages: Sequence[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    raise ValueError("Each training row must contain a user message.")


class RubricAnswerReward(ORM):
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
        messages: Sequence[Sequence[dict[str, Any]]],
        rubrics: Sequence[Sequence[str]],
        prompt_id: Sequence[str],
        **_: Any,
    ) -> list[float]:
        size = len(completions)
        fields = {"messages": messages, "rubrics": rubrics, "prompt_id": prompt_id}
        for name, values in fields.items():
            if len(values) != size:
                raise ValueError(f"{name} must contain one value per completion.")

        groups: dict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(prompt_id):
            groups[str(value)].append(index)

        rewards = [0.0] * size
        for indices in groups.values():
            first = indices[0]
            if len(rubrics[first]) != 1 or not isinstance(rubrics[first][0], str):
                raise ValueError("Each training row must contain one rubric text block.")

            response = self.session.post(
                self.endpoint,
                json={
                    "prompt": user_prompt(messages[first]),
                    "rubric": rubrics[first][0],
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


orms["rubric_answer_reward"] = RubricAnswerReward
