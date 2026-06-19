from __future__ import annotations

import os
from statistics import fmean
from typing import Any, Literal

import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from scoring import judge_prompt, score_judgement


class ScoreRequest(BaseModel):
    prompt: str
    chosen: str
    rejected: str
    rubric: str
    tag: Literal["chosen", "rejected"]
    candidate_scores: list[float]
    candidate_valid: list[bool]


class ScoreResponse(BaseModel):
    rewards: list[float]
    baseline_score: float
    baseline_valid: bool


class PolicyClient:
    def __init__(self) -> None:
        host = os.getenv("POLICY_HOST", "127.0.0.1")
        port = os.getenv("POLICY_PORT", "8010")
        self.endpoint = f"http://{host}:{port}/infer/"
        self.timeout = float(os.getenv("POLICY_TIMEOUT", "300"))
        self.max_tokens = int(os.getenv("JUDGE_MAX_TOKENS", "1024"))
        self.temperature = float(os.getenv("JUDGE_TEMPERATURE", "1.0"))
        self.top_p = float(os.getenv("JUDGE_TOP_P", "0.95"))
        self.top_logprobs = int(os.getenv("JUDGE_TOP_LOGPROBS", "5"))
        self.baseline_samples = int(os.getenv("BASELINE_SAMPLES", "6"))
        self.session = requests.Session()

    def score_baseline(self, prompt: str, response: str, rubric: str) -> tuple[float, bool]:
        conversation = f"user: {prompt}\n\nassistant: {response}"
        request_prompt = judge_prompt(conversation, rubric)
        infer_requests = [
            {"messages": [{"role": "user", "content": request_prompt}]}
            for _ in range(self.baseline_samples)
        ]
        response = self.session.post(
            self.endpoint,
            json={
                "infer_requests": infer_requests,
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
        if len(outputs) != self.baseline_samples:
            raise ValueError("Policy server returned an unexpected number of samples.")

        scored = []
        for output in outputs:
            choice = output["response"]["choices"][0]
            scored.append(
                score_judgement(
                    choice["message"]["content"],
                    choice["logprobs"],
                    rubric,
                )
            )

        scores, valid = zip(*scored)
        return fmean(scores), all(valid)


app = FastAPI(title="Rubric Arrow Reward Server")
policy = PolicyClient()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    if len(request.candidate_scores) != len(request.candidate_valid):
        raise ValueError("Candidate score and validity lengths differ.")

    baseline_response = request.rejected if request.tag == "chosen" else request.chosen
    baseline_score, baseline_valid = policy.score_baseline(
        request.prompt,
        baseline_response,
        request.rubric,
    )

    rewards = []
    for candidate_score, candidate_valid in zip(
        request.candidate_scores,
        request.candidate_valid,
    ):
        if not candidate_valid or not baseline_valid:
            rewards.append(-1.0)
        elif request.tag == "chosen":
            rewards.append(1.0 if candidate_score > baseline_score else -1.0)
        else:
            rewards.append(1.0 if baseline_score > candidate_score else -1.0)

    return ScoreResponse(
        rewards=rewards,
        baseline_score=baseline_score,
        baseline_valid=baseline_valid,
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("REWARD_SERVER_HOST", "127.0.0.1"),
        port=int(os.getenv("REWARD_SERVER_PORT", "8000")),
    )
