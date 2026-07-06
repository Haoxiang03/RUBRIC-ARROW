import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from common import (
    build_conversation_text,
    compute_pointwise_reward,
    create_pointwise_judge_prompt,
    parse_json_to_dict,
    rubric_weights_from_items,
    validate_rubric_format,
)


class ScoreRequest(BaseModel):
    prompt: str
    chosen: str
    rejected: str
    completions: List[str]


class ScoreResponse(BaseModel):
    rewards: List[float]
    debug: Dict[str, Any]


class RewardCalculator:
    def __init__(self) -> None:
        self.judge_host = os.environ.get("JUDGE_VLLM_HOST_IP", "127.0.0.1")
        self.judge_port = int(os.environ.get("JUDGE_VLLM_SERVER_PORT", "8011"))
        self.judge_endpoint = f"http://{self.judge_host}:{self.judge_port}/infer/"

        self.max_tokens = int(os.environ.get("REWARD_MAX_NEW_TOKENS", "1024"))
        self._judge_temperature = float(os.environ.get("REWARD_JUDGE_TEMPERATURE", "0.0"))
        self._judge_top_p = float(os.environ.get("REWARD_JUDGE_TOP_P", "1.0"))
        self._judge_workers = int(os.environ.get("REWARD_JUDGE_WORKERS", "4"))
        self._judge_chunk_size = int(os.environ.get("REWARD_JUDGE_CHUNK_SIZE", "8"))
        self._judge_timeout = float(os.environ.get("REWARD_JUDGE_TIMEOUT", "180"))
        self._judge_retries = max(0, int(os.environ.get("REWARD_JUDGE_RETRIES", "2")))
        self._judge_retry_sleep = max(
            0.0, float(os.environ.get("REWARD_JUDGE_RETRY_SLEEP", "1.5"))
        )
        max_inflight = max(1, int(os.environ.get("REWARD_JUDGE_MAX_INFLIGHT", "2")))
        self._judge_infer_semaphore = threading.BoundedSemaphore(value=max_inflight)
        self._debug_judge_details = str(
            os.environ.get("REWARD_DEBUG_JUDGE_DETAILS", "0")
        ).strip().lower() in ("1", "true", "yes", "on")
        self._debug_max_rubrics = max(1, int(os.environ.get("REWARD_DEBUG_MAX_RUBRICS", "8")))

        self._call_count = 0

        print(f"[RewardCalculator] judge endpoint: {self.judge_endpoint}", flush=True)

    @staticmethod
    def _classify_bool_token(token: str) -> Optional[bool]:
        t = (token or "").strip().lower()
        if t in ("true", "false"):
            return t == "true"
        # common variants from tokenizers / generations
        t = t.strip(",.:;!?)(")
        if t in ("true", "false"):
            return t == "true"
        return None

    def _generate_judges(
        self,
        judge_prompts: List[str],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        top_logprobs = int(os.environ.get("REWARD_LOGPROBS", "10"))
        temperature = self._judge_temperature if temperature is None else float(temperature)
        top_p = self._judge_top_p if top_p is None else float(top_p)
        payload = {
            "infer_requests": [{"messages": [{"role": "user", "content": p}]} for p in judge_prompts],
            "request_config": {
                "max_tokens": self.max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": 1,
                "logprobs": True,
                "top_logprobs": top_logprobs,
            },
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self._judge_retries + 1):
            try:
                with self._judge_infer_semaphore:
                    resp = requests.post(
                        self.judge_endpoint,
                        json=payload,
                        timeout=self._judge_timeout,
                        headers={"Connection": "close"},
                    )
                resp.raise_for_status()
                data = resp.json()

                results: List[Dict[str, Any]] = []
                if isinstance(data, list):
                    for item in data:
                        response = item.get("response", {}) if isinstance(item, dict) else {}
                        choices = response.get("choices", []) if isinstance(response, dict) else []
                        if choices:
                            message = choices[0].get("message", {})
                            content_obj = message.get("content", "")
                            # Be robust to different response schemas.
                            if isinstance(content_obj, str):
                                content = content_obj
                            elif isinstance(content_obj, list):
                                parts: List[str] = []
                                for seg in content_obj:
                                    if isinstance(seg, dict):
                                        txt = seg.get("text")
                                        if isinstance(txt, str):
                                            parts.append(txt)
                                content = "".join(parts)
                            else:
                                content = str(content_obj or "")
                            if not content:
                                text_fallback = choices[0].get("text")
                                if isinstance(text_fallback, str):
                                    content = text_fallback
                            content = content.strip()
                            content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
                            probs = self._extract_probs_from_choice(choices[0])
                            results.append({"text": content, "probs": probs})
                        else:
                            if self._debug_judge_details:
                                print(
                                    "[RewardCalculator] judge response has empty choices",
                                    flush=True,
                                )
                            results.append({"text": "", "probs": {}})
                return results
            except Exception as exc:
                last_exc = exc
                if attempt < self._judge_retries:
                    print(
                        f"[RewardCalculator] judge retry {attempt + 1}/{self._judge_retries} "
                        f"for batch={len(judge_prompts)} due to: {exc}",
                        flush=True,
                    )
                    sleep_sec = self._judge_retry_sleep * (attempt + 1)
                    if sleep_sec > 0:
                        time.sleep(sleep_sec)

        print(f"[RewardCalculator] judge API error: {last_exc}", flush=True)
        return [{"text": "", "probs": {}} for _ in judge_prompts]

    def _generate_judges_parallel(
        self,
        judge_prompts: List[str],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not judge_prompts:
            return []

        chunk_size = max(1, self._judge_chunk_size)
        chunks: List[Tuple[int, List[str]]] = []
        for i in range(0, len(judge_prompts), chunk_size):
            chunks.append((i, judge_prompts[i : i + chunk_size]))

        results: List[Dict[str, Any]] = [{"text": "", "probs": {}} for _ in judge_prompts]
        workers = max(1, min(self._judge_workers, len(chunks)))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._generate_judges,
                    chunk,
                    temperature=temperature,
                    top_p=top_p,
                ): (start, len(chunk))
                for start, chunk in chunks
            }
            for future in as_completed(futures):
                start, chunk_len = futures[future]
                try:
                    out = future.result()
                except Exception:
                    out = [{"text": "", "probs": {}} for _ in range(chunk_len)]
                for j, item in enumerate(out):
                    idx = start + j
                    if idx < len(results):
                        results[idx] = item

        return results

    def _extract_probs_from_choice(self, choice: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
        if not isinstance(choice, dict):
            return {}

        logprobs = choice.get("logprobs")
        if isinstance(logprobs, dict) and isinstance(logprobs.get("content"), list):
            return self._extract_probs_from_openai_style(logprobs.get("content"))

        logprobs_steps = logprobs if isinstance(logprobs, list) else []
        tokens = choice.get("tokens") or choice.get("token_texts") or []
        token_ids = choice.get("token_ids") or []
        if logprobs_steps:
            return self._extract_probs_from_steps(logprobs_steps, tokens=tokens, token_ids=token_ids)
        return {}

    def _extract_probs_from_openai_style(self, entries: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
        context_tail = ""
        waiting_value = False
        active_k: Optional[int] = None
        probs_by_index: Dict[int, Dict[str, float]] = {}

        for entry in entries:
            token = str(entry.get("token", "") or "")
            context_tail += token
            if len(context_tail) > 400:
                context_tail = context_tail[-400:]

            if not waiting_value:
                matches = list(re.finditer(r'"criteria_met_(\d+)"\s*:', context_tail))
                if matches:
                    try:
                        k = int(matches[-1].group(1))
                    except Exception:
                        k = None
                    if k is not None and k not in probs_by_index:
                        waiting_value = True
                        active_k = k
                continue

            if waiting_value and active_k is not None:
                top = entry.get("top_logprobs") or []
                true_logp = None
                false_logp = None

                for cand in top:
                    tok = str(cand.get("token", "") or "")
                    lp = cand.get("logprob")
                    if lp is None:
                        continue
                    if tok == " true":
                        true_logp = lp
                    elif tok == " false":
                        false_logp = lp

                actual_tok = token
                actual_lp = entry.get("logprob")
                if actual_lp is not None:
                    if true_logp is None and actual_tok == " true":
                        true_logp = actual_lp
                    if false_logp is None and actual_tok == " false":
                        false_logp = actual_lp

                probs_by_index[active_k] = {
                    "true_prob": float(math.exp(true_logp)) if true_logp is not None else 0.0,
                    "false_prob": float(math.exp(false_logp)) if false_logp is not None else 0.0,
                }

                waiting_value = False
                active_k = None

        return probs_by_index

    def _extract_probs_from_steps(
        self,
        logprobs_steps: List[Any],
        tokens: Optional[List[str]] = None,
        token_ids: Optional[List[int]] = None,
    ) -> Dict[int, Dict[str, float]]:
        context_tail = ""
        waiting_value = False
        active_k: Optional[int] = None
        probs_by_index: Dict[int, Dict[str, float]] = {}

        tokens = tokens or []
        token_ids = token_ids or []

        def _candidates_from_step(step: Any) -> List[Tuple[str, float]]:
            candidates: List[Tuple[str, float]] = []
            if isinstance(step, dict):
                for k, v in step.items():
                    if isinstance(v, dict):
                        tok = v.get("decoded_token") or v.get("token") or (k if isinstance(k, str) else "")
                        lp = v.get("logprob")
                        if tok is not None and lp is not None:
                            candidates.append((str(tok), float(lp)))
                    elif isinstance(v, (int, float)) and isinstance(k, str):
                        candidates.append((k, float(v)))
            elif isinstance(step, list):
                for v in step:
                    if isinstance(v, dict):
                        tok = v.get("decoded_token") or v.get("token") or ""
                        lp = v.get("logprob")
                        if tok and lp is not None:
                            candidates.append((str(tok), float(lp)))
            return candidates

        for i, step in enumerate(logprobs_steps):
            actual_tok = tokens[i] if i < len(tokens) else ""
            context_tail += actual_tok
            if len(context_tail) > 400:
                context_tail = context_tail[-400:]

            if not waiting_value:
                matches = list(re.finditer(r'"criteria_met_(\d+)"\s*:', context_tail))
                if matches:
                    try:
                        k = int(matches[-1].group(1))
                    except Exception:
                        k = None
                    if k is not None and k not in probs_by_index:
                        waiting_value = True
                        active_k = k
                continue

            if waiting_value and active_k is not None:
                candidates = _candidates_from_step(step)
                true_logp = None
                false_logp = None

                for tok, lp in candidates:
                    if tok == " true":
                        true_logp = lp
                    elif tok == " false":
                        false_logp = lp

                if actual_tok:
                    if true_logp is None and actual_tok == " true":
                        for tok, lp in candidates:
                            if tok == actual_tok:
                                true_logp = lp
                                break
                    if false_logp is None and actual_tok == " false":
                        for tok, lp in candidates:
                            if tok == actual_tok:
                                false_logp = lp
                                break

                probs_by_index[active_k] = {
                    "true_prob": float(math.exp(true_logp)) if true_logp is not None else 0.0,
                    "false_prob": float(math.exp(false_logp)) if false_logp is not None else 0.0,
                }

                waiting_value = False
                active_k = None

        return probs_by_index

    def _score_judge_completion(
        self,
        completion: str,
        probs_by_index: Dict[int, Dict[str, float]],
        rubric_count: int,
        weights: List[float],
    ) -> Tuple[float, str]:
        if rubric_count <= 0:
            return 0.0, "rubric_count=0"

        if probs_by_index:
            total = 0.0
            parts: List[str] = []
            for i in range(1, rubric_count + 1):
                prob_info = probs_by_index.get(i, {})
                tp = float(prob_info.get("true_prob", 0.0) or 0.0)
                fp = float(prob_info.get("false_prob", 0.0) or 0.0)
                w = float(weights[i - 1]) if i - 1 < len(weights) else 1.0
                contrib = (tp - fp) * w
                total += contrib
                if i <= self._debug_max_rubrics:
                    parts.append(
                        f"{i}:tp={tp:.4f},fp={fp:.4f},w={w:.1f},contrib={contrib:.4f}"
                    )
            if rubric_count > self._debug_max_rubrics:
                parts.append(f"... +{rubric_count - self._debug_max_rubrics} more")
            detail = f"mode=logprobs total={total:.4f} | " + "; ".join(parts)
            return total, detail

        parsed = parse_json_to_dict(completion)
        if not parsed and self._debug_judge_details:
            preview = completion[:300].replace("\n", "\\n")
            print(
                f"[RewardCalculator] parse_failed completion_preview={preview}",
                flush=True,
            )
        score, detail = compute_pointwise_reward(parsed, rubric_count)
        return score, f"mode=json total={score:.4f} | {detail}"

    def calculate_rewards(self, req: ScoreRequest) -> ScoreResponse:
        self._call_count += 1

        n = len(req.completions)
        rewards: List[float] = [-1.0] * n
        chosen_scores: List[Optional[float]] = [None] * n
        rejected_scores: List[Optional[float]] = [None] * n
        chosen_details: List[Optional[str]] = [None] * n
        rejected_details: List[Optional[str]] = [None] * n
        format_ok: List[bool] = [False] * n

        if not req.prompt or not req.chosen or not req.rejected:
            return ScoreResponse(
                rewards=rewards,
                debug={"error": "missing_prompt_or_answers"},
            )

        judge_prompts: List[str] = []
        valid_jobs: List[Dict[str, Any]] = []

        for idx, completion in enumerate(req.completions):
            rubric_raw = (completion or "").strip()
            ok, rubric_items, reason = validate_rubric_format(rubric_raw)
            if not ok:
                print(f"[RewardCalculator] comp[{idx}] invalid rubric format: {reason}", flush=True)
                continue

            format_ok[idx] = True
            rubric_count = len(rubric_items)
            weights = rubric_weights_from_items(rubric_items)
            rubric_text = "\n".join(rubric_items)

            chosen_conv = build_conversation_text(req.prompt, req.chosen)
            rejected_conv = build_conversation_text(req.prompt, req.rejected)
            chosen_prompt = create_pointwise_judge_prompt(chosen_conv, rubric_text)
            rejected_prompt = create_pointwise_judge_prompt(rejected_conv, rubric_text)
            judge_prompts.append(chosen_prompt)
            judge_prompts.append(rejected_prompt)

            valid_jobs.append(
                {
                    "idx": idx,
                    "rubric_count": rubric_count,
                    "weights": weights,
                }
            )

        print(
            f"[RewardCalculator] call={self._call_count} total={n} valid={len(valid_jobs)}",
            flush=True,
        )

        if not valid_jobs:
            return ScoreResponse(
                rewards=rewards,
                debug={
                    "chosen_scores": chosen_scores,
                    "rejected_scores": rejected_scores,
                    "rubric_format_ok": format_ok,
                },
            )

        judge_items = self._generate_judges_parallel(
            judge_prompts,
            temperature=self._judge_temperature,
            top_p=self._judge_top_p,
        )
        expected = len(valid_jobs) * 2
        if len(judge_items) < expected:
            judge_items = (judge_items + [{"text": "", "probs": {}}] * expected)[:expected]

        for job_i, job in enumerate(valid_jobs):
            base = 2 * job_i
            chosen_item = judge_items[base]
            rejected_item = judge_items[base + 1]

            chosen_score, chosen_detail = self._score_judge_completion(
                completion=chosen_item.get("text", ""),
                probs_by_index=chosen_item.get("probs", {}) or {},
                rubric_count=job["rubric_count"],
                weights=job["weights"],
            )
            rejected_score, rejected_detail = self._score_judge_completion(
                completion=rejected_item.get("text", ""),
                probs_by_index=rejected_item.get("probs", {}) or {},
                rubric_count=job["rubric_count"],
                weights=job["weights"],
            )

            idx = job["idx"]
            chosen_scores[idx] = chosen_score
            rejected_scores[idx] = rejected_score
            chosen_details[idx] = chosen_detail
            rejected_details[idx] = rejected_detail

        rubric_count_by_idx: Dict[int, int] = {
            int(job["idx"]): int(job["rubric_count"]) for job in valid_jobs
        }
        correct_indices: List[int] = []
        for job in valid_jobs:
            idx = job["idx"]
            cs = chosen_scores[idx]
            rs = rejected_scores[idx]

            if cs is None or rs is None or cs <= rs:
                rewards[idx] = -1.0
            else:
                rewards[idx] = 1.0
                correct_indices.append(idx)

        bonus_indices: set[int] = set()
        correct_avg_rubric_count: Optional[float] = None
        if correct_indices:
            correct_counts = [rubric_count_by_idx.get(i, 0) for i in correct_indices]
            correct_avg_rubric_count = sum(correct_counts) / float(len(correct_counts))
            if correct_avg_rubric_count >= 5.0:
                shortest_count = min(correct_counts)
                for i in correct_indices:
                    if rubric_count_by_idx.get(i, 0) == shortest_count:
                        rewards[i] += 0.1
                        bonus_indices.add(i)

        for job in valid_jobs:
            idx = job["idx"]
            cs = chosen_scores[idx]
            rs = rejected_scores[idx]
            rubric_count = rubric_count_by_idx.get(idx, 0)
            bonus = 0.1 if idx in bonus_indices else 0.0
            print(
                f"[RewardCalculator] comp[{idx}] chosen={float(cs or 0.0):.4f} "
                f"rejected={float(rs or 0.0):.4f} "
                f"rubric_count={rubric_count} "
                f"bonus={bonus:.1f} "
                f"reward={rewards[idx]:.4f}",
                flush=True,
            )
            if self._debug_judge_details:
                print(
                    f"[RewardCalculator] comp[{idx}] chosen_detail: {chosen_details[idx]}",
                    flush=True,
                )
                print(
                    f"[RewardCalculator] comp[{idx}] rejected_detail: {rejected_details[idx]}",
                    flush=True,
                )

        if correct_avg_rubric_count is not None:
            print(
                f"[RewardCalculator] correct_count={len(correct_indices)} "
                f"avg_correct_rubrics={correct_avg_rubric_count:.4f} "
                f"bonus_applied_to={sorted(bonus_indices)}",
                flush=True,
            )

        return ScoreResponse(
            rewards=rewards,
            debug={
                "chosen_scores": chosen_scores,
                "rejected_scores": rejected_scores,
                "rubric_format_ok": format_ok,
                "correct_avg_rubric_count": correct_avg_rubric_count,
                "bonus_indices": sorted(bonus_indices),
                "chosen_details": chosen_details if self._debug_judge_details else [],
                "rejected_details": rejected_details if self._debug_judge_details else [],
            },
        )


app = FastAPI()
calculator: Optional[RewardCalculator] = None


def get_calculator() -> RewardCalculator:
    global calculator
    if calculator is None:
        calculator = RewardCalculator()
    return calculator


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    try:
        return get_calculator().calculate_rewards(req)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    import uvicorn

    host = os.environ.get("REWARD_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("REWARD_SERVER_PORT", "8000"))
    print(f"[RewardServer] starting on {host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
