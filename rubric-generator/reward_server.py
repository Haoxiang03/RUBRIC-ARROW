from __future__ import annotations

import os
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from scoring import (
    assign_rewards,
    judge_prompt,
    parse_rubric,
    rubric_weights,
    score_judgement,
)


class ScoreRequest(BaseModel):
    prompt: str
    chosen: str
    rejected: str
    completions: list[str]


class ScoreResponse(BaseModel):
    rewards: list[float]


class JudgeClient:
    def __init__(self) -> None:
        host = os.getenv("JUDGE_HOST", "127.0.0.1")
        port = os.getenv("JUDGE_PORT", "8011")
        self.endpoint = f"http://{host}:{port}/infer/"
        self.timeout = float(os.getenv("JUDGE_TIMEOUT", "300"))
        self.max_tokens = int(os.getenv("JUDGE_MAX_TOKENS", "1024"))
        self.temperature = float(os.getenv("JUDGE_TEMPERATURE", "1.0"))
        self.top_p = float(os.getenv("JUDGE_TOP_P", "0.95"))
        self.top_logprobs = int(os.getenv("JUDGE_TOP_LOGPROBS", "5"))
        self.session = requests.Session()

    def infer(self, prompts: list[str]) -> list[dict[str, Any]]:
        response = self.session.post(
            self.endpoint,
            json={
                "infer_requests": [
                    {"messages": [{"role": "user", "content": prompt}]}
                    for prompt in prompts
                ],
                "request_config": {
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "n": 1,
                    "logprobs": True,
                    "top_logprobs": self.top_logprobs,
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        outputs: list[dict[str, Any]] = response.json()
        if len(outputs) != len(prompts):
            raise ValueError("Judge server returned an unexpected number of outputs.")
        return [output["response"]["choices"][0] for output in outputs]


app = FastAPI(title="Rubric Generator Reward Server")
judge = JudgeClient()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    if not request.prompt or not request.chosen or not request.rejected:
        raise ValueError("Prompt, chosen response, and rejected response are required.")

    jobs = []
    prompts = []
    for index, completion in enumerate(request.completions):
        try:
            items = parse_rubric(completion)
        except ValueError:
            continue

        rubric = "\n".join(items)
        weights = rubric_weights(items)
        jobs.append((index, len(items), weights))
        prompts.extend(
            [
                judge_prompt(request.prompt, request.chosen, rubric),
                judge_prompt(request.prompt, request.rejected, rubric),
            ]
        )

    chosen_scores = [0.0] * len(request.completions)
    rejected_scores = [0.0] * len(request.completions)
    rubric_counts = [0] * len(request.completions)
    valid = [False] * len(request.completions)

    if jobs:
        outputs = judge.infer(prompts)
        for job_index, (completion_index, rubric_count, weights) in enumerate(jobs):
            chosen = outputs[2 * job_index]
            rejected = outputs[2 * job_index + 1]
            chosen_score, chosen_valid = score_judgement(
                chosen["message"]["content"],
                chosen["logprobs"],
                weights,
            )
            rejected_score, rejected_valid = score_judgement(
                rejected["message"]["content"],
                rejected["logprobs"],
                weights,
            )
            chosen_scores[completion_index] = chosen_score
            rejected_scores[completion_index] = rejected_score
            rubric_counts[completion_index] = rubric_count
            valid[completion_index] = chosen_valid and rejected_valid

    return ScoreResponse(
        rewards=assign_rewards(
            chosen_scores,
            rejected_scores,
            rubric_counts,
            valid,
        )
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("REWARD_SERVER_HOST", "127.0.0.1"),
        port=int(os.getenv("REWARD_SERVER_PORT", "8000")),
    )
