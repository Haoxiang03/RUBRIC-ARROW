from __future__ import annotations

import os
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from scoring import judge_prompt, parse_rubrics, rubric_weights, score_judgement


class ScoreRequest(BaseModel):
    prompt: str
    rubric: str
    completions: list[str]


class ScoreResponse(BaseModel):
    rewards: list[float]
    valid: list[bool]


class JudgeClient:
    def __init__(self) -> None:
        host = os.getenv("JUDGE_HOST", "127.0.0.1")
        port = os.getenv("JUDGE_PORT", "8011")
        self.endpoint = f"http://{host}:{port}/infer/"
        self.timeout = float(os.getenv("JUDGE_TIMEOUT", "300"))
        self.max_tokens = int(os.getenv("JUDGE_MAX_TOKENS", "1024"))
        self.temperature = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
        self.top_p = float(os.getenv("JUDGE_TOP_P", "1.0"))
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


app = FastAPI(title="Rubric RLHF Reward Server")
judge = JudgeClient()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    if not request.prompt or not request.completions:
        raise ValueError("Prompt and completions are required.")

    items = parse_rubrics(request.rubric)
    weights = rubric_weights(items)
    rubric = "\n".join(items)
    outputs = judge.infer(
        [
            judge_prompt(request.prompt, completion, rubric)
            for completion in request.completions
        ]
    )

    scored = [
        score_judgement(
            output["message"]["content"],
            output["logprobs"],
            weights,
        )
        for output in outputs
    ]
    rewards, valid = zip(*scored)
    return ScoreResponse(rewards=list(rewards), valid=list(valid))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("REWARD_SERVER_HOST", "127.0.0.1"),
        port=int(os.getenv("REWARD_SERVER_PORT", "8000")),
    )
