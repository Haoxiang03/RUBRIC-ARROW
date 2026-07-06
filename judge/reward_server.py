import os
import re
import math
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from common import (
    build_conversation_text,
    count_rubric_items,
    create_pointwise_judge_prompt,
    format_is_valid,
    parse_json_to_dict,
)


class ScoreRequest(BaseModel):
    prompt: str
    chosen: str
    rejected: str
    rubrics: List[str]
    completions: List[str]
    completion_probs: Optional[List[Any]] = None
    completion_scores: Optional[List[float]] = None
    completion_fmts: Optional[List[bool]] = None
    tag: str


class ScoreResponse(BaseModel):
    rewards: List[float]
    debug: Dict[str, Any]


class RewardCalculator:
    def __init__(self):
        self.vllm_host = os.environ.get("VLLM_HOST_IP", "127.0.0.1")
        self.vllm_port = int(os.environ.get("VLLM_SERVER_PORT", "8010"))
        self.vllm_endpoint = f"http://{self.vllm_host}:{self.vllm_port}/infer/"
        self.max_tokens = int(os.environ.get("REWARD_MAX_NEW_TOKENS", "512"))
        self._judge_temperature = float(os.environ.get("REWARD_JUDGE_TEMPERATURE", "1.0"))
        self._judge_top_p = float(os.environ.get("REWARD_JUDGE_TOP_P", "0.95"))
        
        self._session = requests.Session()
        self._call_count = 0
        self._judge_workers = int(os.environ.get("REWARD_JUDGE_WORKERS", "4"))
        self._judge_chunk_size = int(os.environ.get("REWARD_JUDGE_CHUNK_SIZE", "8"))
        self._require_probs = True
        self._debug_topk = str(os.environ.get("REWARD_DEBUG_TOPK", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        
        print(f"[RewardCalculator] vLLM endpoint: {self.vllm_endpoint}", flush=True)
        print(
            f"[RewardCalculator] judge sampling: temperature={self._judge_temperature}, top_p={self._judge_top_p}",
            flush=True,
        )
    
    def _generate_judges(
        self,
        judge_prompts: List[str],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        try:
            top_logprobs = int(os.environ.get("REWARD_LOGPROBS", "10"))
            temperature = self._judge_temperature if temperature is None else float(temperature)
            top_p = self._judge_top_p if top_p is None else float(top_p)
            resp = self._session.post(
                self.vllm_endpoint,
                json={
                    "infer_requests": [
                        {"messages": [{"role": "user", "content": p}]} for p in judge_prompts
                    ],
                    "request_config": {
                        "max_tokens": self.max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "n": 1,
                        "logprobs": True,
                        "top_logprobs": top_logprobs,
                    }
                },
                timeout=120,
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
                        content = message.get("content", "").strip()
                        content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
                        probs = self._extract_probs_from_choice(choices[0])
                        results.append({"text": content.strip(), "probs": probs})
                    else:
                        results.append({"text": "", "probs": {}})
            return results
            
        except Exception as e:
            print(f"[RewardCalculator] vLLM API error: {e}", flush=True)
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
            chunks.append((i, judge_prompts[i:i + chunk_size]))

        results: List[Dict[str, Any]] = [{"text": "", "probs": {}} for _ in judge_prompts]
        workers = max(1, min(self._judge_workers, len(chunks)))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(self._generate_judges, ch, temperature=temperature, top_p=top_p): (start, len(ch))
                for start, ch in chunks
            }
            for fut in as_completed(futs):
                start, ch_len = futs[fut]
                try:
                    out = fut.result()
                except Exception:
                    out = [{"text": "", "probs": {}} for _ in range(ch_len)]
                for j, item in enumerate(out):
                    if start + j < len(results):
                        results[start + j] = item

        return results

    def _weight_from_tags(self, tags: List[str]) -> float:
        tags = [str(t).strip().lower() for t in (tags or [])]
        if "constraint" in tags:
            return 5.0
        if "hard rule" in tags:
            return 3.0
        if "principle" in tags:
            return 1.0
        return 1.0

    def _weights_from_rubric_text(self, rubric_text: str, rubric_count: int) -> List[float]:
        weights = [1.0] * rubric_count
        pattern = re.compile(r"(?m)^\s*(\d+)\.\s*.*?(?:\[(.*?)\])?\s*$")
        for m in pattern.finditer(rubric_text or ""):
            try:
                idx = int(m.group(1))
            except Exception:
                continue
            if idx < 1 or idx > rubric_count:
                continue
            bracket = (m.group(2) or "").strip()
            tags = [t.strip() for t in bracket.split(",") if t.strip()] if bracket else []
            weights[idx - 1] = self._weight_from_tags(tags)
        return weights

    def _extract_probs_from_choice(self, choice: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
        if not isinstance(choice, dict):
            return {}

        lp = choice.get("logprobs")
        if isinstance(lp, dict) and isinstance(lp.get("content"), list):
            return self._extract_probs_from_openai_style(lp.get("content"))

        logprobs_steps = lp if isinstance(lp, list) else []
        tokens = choice.get("tokens") or choice.get("token_texts") or []
        token_ids = choice.get("token_ids") or []
        if logprobs_steps:
            return self._extract_probs_from_steps(logprobs_steps, tokens=tokens, token_ids=token_ids)

        return {}

    @staticmethod
    def _normalize_probs_by_index(raw: Any) -> Dict[int, Dict[str, float]]:
        if not isinstance(raw, dict):
            return {}
        out: Dict[int, Dict[str, float]] = {}
        for key, value in raw.items():
            try:
                idx = int(key)
            except Exception:
                continue
            if not isinstance(value, dict):
                continue
            tp = value.get("true_prob", 0.0)
            fp = value.get("false_prob", 0.0)
            try:
                out[idx] = {
                    "true_prob": float(tp or 0.0),
                    "false_prob": float(fp or 0.0),
                }
            except Exception:
                continue
        return out

    def _extract_probs_from_completion_payload(self, payload: Any) -> Dict[int, Dict[str, float]]:
        if payload is None:
            return {}

        direct = self._normalize_probs_by_index(payload)
        if direct:
            return direct

        if isinstance(payload, dict):
            wrapped = self._normalize_probs_by_index(payload.get("probs"))
            if wrapped:
                return wrapped

            logprobs = payload.get("logprobs")
            if isinstance(logprobs, dict) and isinstance(logprobs.get("content"), list):
                return self._extract_probs_from_openai_style(logprobs.get("content"))

            if isinstance(logprobs, list):
                tokens = payload.get("tokens") or payload.get("token_texts") or []
                token_ids = payload.get("token_ids") or []
                return self._extract_probs_from_steps(logprobs, tokens=tokens, token_ids=token_ids)

            content = payload.get("content")
            if isinstance(content, list):
                return self._extract_probs_from_openai_style(content)

        if isinstance(payload, list):
            if payload and isinstance(payload[0], dict):
                first = payload[0]
                if "token" in first or "top_logprobs" in first:
                    return self._extract_probs_from_openai_style(payload)
                return self._extract_probs_from_steps(payload, tokens=[], token_ids=[])

        return {}

    def _extract_probs_from_openai_style(self, entries: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
        context_tail = ""
        waiting_value = False
        active_k: Optional[int] = None
        probs_by_index: Dict[int, Dict[str, float]] = {}

        for i, entry in enumerate(entries):
            token = str(entry.get("token", "") or "")
            context_tail += token
            if len(context_tail) > 400:
                context_tail = context_tail[-400:]

            if not waiting_value:
                matches = list(re.finditer(r'"criteria_met_(\d+)"\s*:', context_tail))
                if matches:
                    last_match = matches[-1]
                    try:
                        k = int(last_match.group(1))
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

                if self._debug_topk:
                    print(f"\n===== criteria_met_{active_k} step top-k (token index {i}) =====", flush=True)
                    for cand in top[:10]:
                        tok = str(cand.get("token", "") or "")
                        lp = cand.get("logprob")
                        p = math.exp(lp) if lp is not None else 0.0
                        lp_str = f"{lp:.6f}" if lp is not None else "None"
                        print(f"  token={repr(tok)}  logp={lp_str}  prob={p:.6e}", flush=True)
                    print("===== END =====\n", flush=True)

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

                true_prob = math.exp(true_logp) if true_logp is not None else 0.0
                false_prob = math.exp(false_logp) if false_logp is not None else 0.0

                probs_by_index[active_k] = {
                    "true_prob": float(true_prob),
                    "false_prob": float(false_prob),
                }

                waiting_value = False
                active_k = None

        return probs_by_index

    def _extract_probs_from_steps(
        self,
        logprobs_steps: List[Any],
        tokens: List[str] | None = None,
        token_ids: List[int] | None = None,
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
                    last_match = matches[-1]
                    try:
                        k = int(last_match.group(1))
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

                if self._debug_topk:
                    print(f"\n===== criteria_met_{active_k} step top-k (token index {i}) =====", flush=True)
                    for tok, lp in candidates[:10]:
                        p = math.exp(lp) if lp is not None else 0.0
                        lp_str = f"{lp:.6f}" if lp is not None else "None"
                        print(f"  token={repr(tok)}  logp={lp_str}  prob={p:.6e}", flush=True)
                    print("===== END =====\n", flush=True)

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

                true_prob = math.exp(true_logp) if true_logp is not None else 0.0
                false_prob = math.exp(false_logp) if false_logp is not None else 0.0

                probs_by_index[active_k] = {
                    "true_prob": float(true_prob),
                    "false_prob": float(false_prob),
                }

                waiting_value = False
                active_k = None

        return probs_by_index
    
    def _parse_completion(
        self,
        completion: str,
        rubric_count: int,
        label: str = "",
        probs_by_index: Optional[Dict[int, Dict[str, float]]] = None,
        weights: Optional[List[float]] = None,
    ) -> tuple[float, bool, str]:
        if not completion or not completion.strip():
            return 0.0, False, "empty"

        parsed = parse_json_to_dict(completion)
        if not parsed:
            preview = completion[:200] if len(completion) > 200 else completion
            print(f"    [{label}] parse_failed, content preview: {preview}", flush=True)
            return 0.0, False, "parse_failed"

        probs_by_index = probs_by_index or {}
        use_probs = bool(probs_by_index)
        if weights is None or len(weights) != rubric_count:
            weights = [1.0] * rubric_count

        if self._require_probs and not use_probs:
            print(
                f"    [{label}] missing_probs -> probability-only scoring requires completion_probs/logprobs",
                flush=True,
            )
            return 0.0, False, "missing_probs"

        total = 0.0
        details = []
        prob_covered = 0

        for i in range(1, rubric_count + 1):
            prob_info = probs_by_index.get(i, {})
            if i in probs_by_index:
                prob_covered += 1
            tp = float(prob_info.get("true_prob", 0.0) or 0.0)
            fp = float(prob_info.get("false_prob", 0.0) or 0.0)
            w = float(weights[i - 1])
            step = (tp - fp) * w
            total += step
            details.append(f"[{i}]{step:.4f}")

        score = total
        detail_str = (
            f"mode=probs covered={prob_covered}/{rubric_count} "
            + "+".join(details)
            + f"={score:.4f}"
        )

        fmt_ok = format_is_valid(parsed, rubric_count) and (prob_covered == rubric_count)

        print(f"    [{label}] Judge score calculation:", flush=True)
        running = 0.0
        for i in range(1, rubric_count + 1):
            criteria_met = parsed.get(f"criteria_met_{i}", "N/A")
            explanation = parsed.get(f"explanation_{i}", "N/A")
            if isinstance(explanation, str) and len(explanation) > 80:
                explanation = explanation[:80] + "..."
            prob_info = probs_by_index.get(i, {})
            tp = float(prob_info.get("true_prob", 0.0) or 0.0)
            fp = float(prob_info.get("false_prob", 0.0) or 0.0)
            w = float(weights[i - 1]) if weights else 1.0
            step = (tp - fp) * w
            running += step
            print(
                f"      criteria_{i}: {criteria_met} | tp={tp:.4f} fp={fp:.4f} w={w:.1f} step={step:+.4f} sum={running:+.4f} | {explanation}",
                flush=True,
            )
        print(f"    [{label}] Final: score={score:.4f}, fmt_ok={fmt_ok}", flush=True)

        return score, fmt_ok, detail_str
    
    def calculate_rewards(self, req: ScoreRequest) -> ScoreResponse:
        self._call_count += 1
        
        rubric_text = "\n".join([r.strip() for r in req.rubrics if r.strip()])
        rubric_count = count_rubric_items(rubric_text)
        weights = self._weights_from_rubric_text(rubric_text, rubric_count)
        
        print(f"\n[RewardCalculator] Call #{self._call_count}: tag={req.tag}, rubric_count={rubric_count}, k={len(req.completions)}", flush=True)
        
        if rubric_count == 0:
            return ScoreResponse(rewards=[0.0] * len(req.completions), debug={"error": "rubric_count=0"})
        
        if req.tag.lower() == "rejected":
            baseline_response = req.chosen
            baseline_label = "chosen"
        else:
            baseline_response = req.rejected
            baseline_label = "rejected"
        
        baseline_samples = 6

        baseline_conversation = build_conversation_text(req.prompt, baseline_response)
        baseline_judge_prompt = create_pointwise_judge_prompt(baseline_conversation, rubric_text)
        baseline_judge_prompts = [baseline_judge_prompt for _ in range(baseline_samples)]

        judge_items = self._generate_judges_parallel(
            baseline_judge_prompts,
            temperature=self._judge_temperature,
            top_p=self._judge_top_p,
        )
        expected_count = baseline_samples
        if len(judge_items) < expected_count:
            judge_items = (judge_items + [{"text": "", "probs": {}}] * expected_count)[:expected_count]

        baseline_scores = []
        baseline_fmts = []
        for i, baseline_item in enumerate(judge_items[:baseline_samples]):
            baseline_text = baseline_item.get("text", "")
            baseline_probs = baseline_item.get("probs", {}) or {}
            b_score, b_fmt, _ = self._parse_completion(
                baseline_text,
                rubric_count,
                label=f"baseline({baseline_label})[{i}]",
                probs_by_index=baseline_probs,
                weights=weights,
            )
            baseline_scores.append(b_score)
            baseline_fmts.append(b_fmt)

        valid_baseline_scores = [s for s, ok in zip(baseline_scores, baseline_fmts) if ok]
        if valid_baseline_scores:
            baseline_score = sum(valid_baseline_scores) / len(valid_baseline_scores)
            baseline_fmt = True
        else:
            baseline_score = (sum(baseline_scores) / len(baseline_scores)) if baseline_scores else 0.0
            baseline_fmt = False

        print(
            f"  baseline_avg={baseline_score:.4f} from {len(valid_baseline_scores)}/{len(baseline_scores)} valid samples",
            flush=True,
        )
        
        rewards = []
        comp_scores = []
        comp_fmts = []
        completion_scores = req.completion_scores or []
        completion_fmts = req.completion_fmts or []
        use_precomputed = len(completion_scores) == len(req.completions)
        print(
            f"  completion_scores provided={len(completion_scores)}/{len(req.completions)} "
            f"completion_fmts={len(completion_fmts)}/{len(req.completions)} "
            f"use_precomputed={use_precomputed}",
            flush=True,
        )

        if use_precomputed:
            for i, comp in enumerate(req.completions):
                try:
                    comp_score = float(completion_scores[i])
                except Exception:
                    comp_score = 0.0
                comp_fmt = bool(completion_fmts[i]) if i < len(completion_fmts) else False
                comp_scores.append(comp_score)
                comp_fmts.append(comp_fmt)

                format_ok = comp_fmt and baseline_fmt

                if not format_ok:
                    reward = -1.0
                    reason = "fmt_err"
                elif req.tag.lower() == "chosen":
                    if comp_score > baseline_score:
                        reward = 1.0
                        reason = f"chosen({comp_score:.3f})>rejected({baseline_score:.3f})"
                    else:
                        reward = -1.0
                        reason = f"chosen({comp_score:.3f})<=rejected({baseline_score:.3f})"
                else:
                    if baseline_score > comp_score:
                        reward = 1.0
                        reason = f"chosen({baseline_score:.3f})>rejected({comp_score:.3f})"
                    else:
                        reward = -1.0
                        reason = f"chosen({baseline_score:.3f})<=rejected({comp_score:.3f})"

                rewards.append(reward)
                print(
                    f"  => comp[{i}] precomputed_score={comp_score:.4f} fmt={comp_fmt} "
                    f"reward={reward:.0f} ({reason})",
                    flush=True,
                )
        else:
            completion_probs = req.completion_probs or []
            provided_non_null = sum(1 for p in completion_probs if p is not None)
            print(
                f"  completion_probs provided={len(completion_probs)}/{len(req.completions)} "
                f"(non_null={provided_non_null})",
                flush=True,
            )

            for i, comp in enumerate(req.completions):
                comp_payload = completion_probs[i] if i < len(completion_probs) else None
                comp_probs = self._extract_probs_from_completion_payload(comp_payload)
                if not comp_probs:
                    payload_type = type(comp_payload).__name__
                    payload_keys = []
                    if isinstance(comp_payload, dict):
                        payload_keys = sorted([str(k) for k in comp_payload.keys()])[:8]
                    print(
                        f"  comp[{i}] extracted_probs=0/{rubric_count} "
                        f"payload_type={payload_type} payload_keys={payload_keys}",
                        flush=True,
                    )
                comp_score, comp_fmt, _ = self._parse_completion(
                    comp, rubric_count, label=f"comp[{i}]", probs_by_index=comp_probs, weights=weights
                )
                comp_scores.append(comp_score)
                comp_fmts.append(comp_fmt)

                format_ok = comp_fmt and baseline_fmt

                if not format_ok:
                    reward = -1.0
                    reason = "fmt_err"
                elif req.tag.lower() == "chosen":
                    if comp_score > baseline_score:
                        reward = 1.0
                        reason = f"chosen({comp_score:.3f})>rejected({baseline_score:.3f})"
                    else:
                        reward = -1.0
                        reason = f"chosen({comp_score:.3f})<=rejected({baseline_score:.3f})"
                else:
                    if baseline_score > comp_score:
                        reward = 1.0
                        reason = f"chosen({baseline_score:.3f})>rejected({comp_score:.3f})"
                    else:
                        reward = -1.0
                        reason = f"chosen({baseline_score:.3f})<=rejected({comp_score:.3f})"

                rewards.append(reward)
                print(f"  => comp[{i}] reward={reward:.0f} ({reason})", flush=True)

        print(f"\n  Summary: rewards={rewards}", flush=True)
        
        return ScoreResponse(
            rewards=rewards,
            debug={
                "baseline_score": baseline_score,
                "baseline_scores": baseline_scores,
                "baseline_fmt": baseline_fmt,
                "baseline_fmts": baseline_fmts,
                "comp_scores": comp_scores,
                "comp_fmts": comp_fmts,
            }
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
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    import uvicorn
    host = os.environ.get("REWARD_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("REWARD_SERVER_PORT", "8000"))
    print(f"[RewardServer] Starting on {host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
