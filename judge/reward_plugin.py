from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Sequence

import requests

from scoring import score_judgement
from swift.plugin import ORM, orms
from swift.trainers.rlhf_trainer.rollout_mixin import RolloutTrainerMixin


def enable_rollout_logprobs() -> None:
    if getattr(RolloutTrainerMixin, "_rubric_arrow_logprob_patch", False):
        return

    prepare = RolloutTrainerMixin._prepare_rollout_params
    postprocess = RolloutTrainerMixin._postprocess_rollout_outputs

    def prepare_with_logprobs(self, *args, **kwargs):
        prepare(self, *args, **kwargs)
        self.request_config.logprobs = True
        self.request_config.top_logprobs = int(os.getenv("ROLLOUT_TOP_LOGPROBS", "5"))

    def postprocess_with_logprobs(self, inputs, outputs):
        rows = postprocess(self, inputs, outputs)
        if len(rows) != len(outputs):
            raise ValueError("Rollout rows and outputs have different lengths.")
        for row, output in zip(rows, outputs):
            row["completion_logprobs"] = output.response.choices[0].logprobs
        return rows

    RolloutTrainerMixin._prepare_rollout_params = prepare_with_logprobs
    RolloutTrainerMixin._postprocess_rollout_outputs = postprocess_with_logprobs
    RolloutTrainerMixin._rubric_arrow_logprob_patch = True


enable_rollout_logprobs()


def rubric_text(value: str | Sequence[str]) -> str:
    if isinstance(value, str):
        return value.strip()
    return "\n".join(item.strip() for item in value if item.strip())


class PairwiseJudgeReward(ORM):
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
        rubrics: Sequence[str | Sequence[str]],
        tag: Sequence[str],
        original_prompt: Sequence[str],
        prompt_id: Sequence[str],
        completion_logprobs: Sequence[dict[str, Any]],
        **_: Any,
    ) -> list[float]:
        size = len(completions)
        fields = {
            "chosen": chosen,
            "rejected": rejected,
            "rubrics": rubrics,
            "tag": tag,
            "original_prompt": original_prompt,
            "prompt_id": prompt_id,
            "completion_logprobs": completion_logprobs,
        }
        for name, values in fields.items():
            if len(values) != size:
                raise ValueError(f"{name} must contain one value per completion.")

        groups: dict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(prompt_id):
            groups[str(value)].append(index)

        rewards = [0.0] * size
        for indices in groups.values():
            first = indices[0]
            rubric = rubric_text(rubrics[first])
            scores, valid = zip(
                *[
                    score_judgement(completions[index], completion_logprobs[index], rubric)
                    for index in indices
                ]
            )

            response = self.session.post(
                self.endpoint,
                json={
                    "prompt": original_prompt[first],
                    "chosen": chosen[first],
                    "rejected": rejected[first],
                    "rubric": rubric,
                    "tag": tag[first],
                    "candidate_scores": scores,
                    "candidate_valid": valid,
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


orms["pairwise_judge_reward"] = PairwiseJudgeReward
