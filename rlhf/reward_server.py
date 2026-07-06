import math
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from common import (
    build_conversation_text,
    compute_pointwise_reward_from_json,
    create_pointwise_judge_prompt,
    normalize_rubrics,
    parse_json_to_dict,
    rubric_weights_from_items,
)


class ScoreRequest(BaseModel):
    prompt: str
    rubrics: List[Any]
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
        self._judge_retry_sleep = max(0.0, float(os.environ.get("REWARD_JUDGE_RETRY_SLEEP", "1.5")))
        self._debug_judge_details = str(os.environ.get("REWARD_DEBUG_JUDGE_DETAILS", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._debug_max_rubrics = max(1, int(os.environ.get("REWARD_DEBUG_MAX_RUBRICS", "6")))

        max_inflight = max(1, int(os.environ.get("REWARD_JUDGE_MAX_INFLIGHT", "2")))
        self._judge_infer_semaphore = threading.BoundedSemaphore(value=max_inflight)
        self._judge_log_lock = threading.Lock()

        judge_log_path = str(os.environ.get("REWARD_JUDGE_LOG_PATH", "")).strip()
        self._judge_log_path = os.path.abspath(judge_log_path) if judge_log_path else ""
        self._judge_log_enabled = bool(self._judge_log_path)
        if self._judge_log_enabled:
            log_dir = os.path.dirname(self._judge_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

        self._call_count = 0

        print(f"[RewardCalculator] judge endpoint: {self.judge_endpoint}", flush=True)
        if self._judge_log_enabled:
            print(f"[RewardCalculator] judge log path: {self._judge_log_path}", flush=True)
        else:
            print("[RewardCalculator] judge log path: <disabled>", flush=True)

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
                            results.append({"text": "", "probs": {}})
                return results
            except Exception as exc:
                last_exc = exc
                if attempt < self._judge_retries:
                    sleep_sec = self._judge_retry_sleep * (attempt + 1)
                    print(
                        f"[RewardCalculator] judge retry {attempt + 1}/{self._judge_retries} "
                        f"for batch={len(judge_prompts)} due to: {exc}",
                        flush=True,
                    )
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

    def _score_completion(
        self,
        completion: str,
        probs_by_index: Dict[int, Dict[str, float]],
        rubric_count: int,
        weights: List[float],
        parsed_judge: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str]:
        if rubric_count <= 0:
            return 0.0, "rubric_count=0"

        if probs_by_index:
            total = 0.0
            total_weight = 0.0
            parts: List[str] = []
            for i in range(1, rubric_count + 1):
                prob_info = probs_by_index.get(i, {})
                tp = float(prob_info.get("true_prob", 0.0) or 0.0)
                fp = float(prob_info.get("false_prob", 0.0) or 0.0)
                w = float(weights[i - 1]) if i - 1 < len(weights) else 1.0
                total_weight += w
                contrib = (tp - fp) * w
                total += contrib
                if i <= self._debug_max_rubrics:
                    parts.append(f"{i}:tp={tp:.4f},fp={fp:.4f},w={w:.1f},s={contrib:.4f}")
            if total_weight <= 0:
                return 0.0, "mode=logprobs weight_sum<=0"
            normalized = total / total_weight
            if rubric_count > self._debug_max_rubrics:
                parts.append(f"... +{rubric_count - self._debug_max_rubrics} more")
            return normalized, f"mode=logprobs raw={total:.4f} weight_sum={total_weight:.4f} norm={normalized:.4f} | " + "; ".join(parts)

        parsed = parsed_judge if isinstance(parsed_judge, dict) else parse_json_to_dict(completion)
        score, detail = compute_pointwise_reward_from_json(parsed, rubric_count, weights)
        return score, f"mode=json | {detail}"

    @staticmethod
    def _as_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _to_optional_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "yes", "1"}:
                return True
            if text in {"false", "no", "0"}:
                return False
        return None

    @staticmethod
    def _to_optional_prob(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "yes"}:
                return 1.0
            if text in {"false", "no"}:
                return 0.0
            try:
                return max(0.0, min(1.0, float(text)))
            except Exception:
                return None
        return None

    def _build_rubric_log_entries(
        self,
        parsed_judge: Dict[str, Any],
        probs_by_index: Dict[int, Dict[str, float]],
        rubric_items: Sequence[str],
        rubric_count: int,
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for i in range(1, rubric_count + 1):
            prob_info = probs_by_index.get(i, {}) if isinstance(probs_by_index, dict) else {}
            true_prob = self._to_optional_prob(prob_info.get("true_prob"))
            false_prob = self._to_optional_prob(prob_info.get("false_prob"))

            if true_prob is None:
                true_prob = self._to_optional_prob(parsed_judge.get(f"true_prob_{i}"))
            if false_prob is None:
                false_prob = self._to_optional_prob(parsed_judge.get(f"false_prob_{i}"))

            criteria_met = self._to_optional_bool(parsed_judge.get(f"criteria_met_{i}"))
            if criteria_met is None and true_prob is not None and false_prob is not None:
                criteria_met = true_prob >= false_prob
            if true_prob is None and criteria_met is not None:
                true_prob = 1.0 if criteria_met else 0.0
            if false_prob is None and criteria_met is not None:
                false_prob = 0.0 if criteria_met else 1.0

            explanation = self._as_text(parsed_judge.get(f"explanation_{i}"))
            rubric_text = self._as_text(rubric_items[i - 1]) if i - 1 < len(rubric_items) else ""

            entries.append(
                {
                    "rubric_index": i,
                    "rubric": rubric_text,
                    "true_prob": true_prob,
                    "false_prob": false_prob,
                    "criteria_met": criteria_met,
                    "explanation": explanation,
                }
            )
        return entries

    def _append_judge_log(self, payload: Dict[str, Any]) -> None:
        if not self._judge_log_enabled:
            return
        try:
            line = json.dumps(payload, ensure_ascii=False)
            with self._judge_log_lock:
                with open(self._judge_log_path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.write("\n")
        except Exception as exc:
            print(f"[RewardCalculator] judge log write error: {exc}", flush=True)

    def calculate_rewards(self, req: ScoreRequest) -> ScoreResponse:
        self._call_count += 1

        n = len(req.completions)
        rewards: List[float] = [0.0] * n
        score_details: List[str] = [""] * n

        if not req.prompt or not req.completions:
            return ScoreResponse(rewards=rewards, debug={"error": "missing_prompt_or_completions"})

        rubric_items = normalize_rubrics(req.rubrics)
        rubric_count = len(rubric_items)
        if rubric_count <= 0:
            return ScoreResponse(rewards=rewards, debug={"error": "missing_or_invalid_rubrics"})

        weights = rubric_weights_from_items(rubric_items)
        rubric_text = "\n".join(rubric_items)

        judge_prompts = [
            create_pointwise_judge_prompt(build_conversation_text(req.prompt, comp or ""), rubric_text)
            for comp in req.completions
        ]

        print(
            f"[RewardCalculator] call={self._call_count} n={n} rubric_count={rubric_count} "
            f"weight_sum={float(sum(weights)):.4f}",
            flush=True,
        )
        if self._debug_judge_details:
            print(f"[RewardCalculator] weights={weights}", flush=True)

        judge_items = self._generate_judges_parallel(
            judge_prompts,
            temperature=self._judge_temperature,
            top_p=self._judge_top_p,
        )
        if len(judge_items) < n:
            judge_items = (judge_items + [{"text": "", "probs": {}}] * n)[:n]

        for idx, item in enumerate(judge_items):
            judge_text = item.get("text", "") or ""
            parsed_judge = parse_json_to_dict(judge_text)
            probs_by_index = item.get("probs", {}) or {}
            rubric_entries = self._build_rubric_log_entries(
                parsed_judge=parsed_judge,
                probs_by_index=probs_by_index,
                rubric_items=rubric_items,
                rubric_count=rubric_count,
            )
            score, detail = self._score_completion(
                completion=judge_text,
                probs_by_index=probs_by_index,
                rubric_count=rubric_count,
                weights=weights,
                parsed_judge=parsed_judge,
            )
            rewards[idx] = float(score)
            score_details[idx] = detail

            self._append_judge_log(
                {
                    "ts_unix": time.time(),
                    "call": self._call_count,
                    "completion_index": idx,
                    "prompt": req.prompt,
                    "completion": req.completions[idx] if idx < len(req.completions) else "",
                    "judge_prompt": judge_prompts[idx] if idx < len(judge_prompts) else "",
                    "judge_text": judge_text,
                    "judge_parsed": parsed_judge,
                    "rubric_results": rubric_entries,
                    "reward": float(score),
                    "reward_detail": detail,
                }
            )

            if self._debug_judge_details:
                print(f"[RewardCalculator] comp[{idx}] reward={score:.4f} detail={detail}", flush=True)

        return ScoreResponse(
            rewards=rewards,
            debug={
                "rubric_count": rubric_count,
                "weights": weights,
                "weight_sum": float(sum(weights)),
                "details": score_details if self._debug_judge_details else [],
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
