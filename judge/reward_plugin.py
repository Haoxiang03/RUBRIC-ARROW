import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from common import count_rubric_items, format_is_valid, parse_json_to_dict
from swift.plugin import ORM, orms


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name, "1" if default else "0")
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _patch_rollout_logprobs() -> None:
    try:
        from swift.trainers.rlhf_trainer.rollout_mixin import RolloutTrainerMixin
    except Exception as e:
        print(f"[PointwiseJudgeReward] rollout patch skipped (import failed): {e}", flush=True)
        return

    if getattr(RolloutTrainerMixin, "_openrubrics_logprob_patch", False):
        return

    orig_prepare = RolloutTrainerMixin._prepare_rollout_params
    orig_postprocess = RolloutTrainerMixin._postprocess_rollout_outputs

    def _prepare_rollout_params_patched(self, *args, **kwargs):
        orig_prepare(self, *args, **kwargs)
        if not _env_bool("GRPO_ROLLOUT_LOGPROBS", True):
            return
        try:
            top_logprobs = int(os.environ.get("GRPO_ROLLOUT_TOP_LOGPROBS", "10"))
            if top_logprobs < 1:
                top_logprobs = 1
            self.request_config.logprobs = True
            self.request_config.top_logprobs = top_logprobs
            print(
                f"[PointwiseJudgeReward] rollout request logprobs enabled: top_logprobs={top_logprobs}",
                flush=True,
            )
        except Exception as e:
            print(f"[PointwiseJudgeReward] rollout logprobs setup failed: {e}", flush=True)

    def _postprocess_rollout_outputs_patched(self, inputs, outputs):
        processed = orig_postprocess(self, inputs, outputs)
        try:
            if not isinstance(processed, list) or not isinstance(outputs, list):
                return processed
            if len(processed) == len(outputs):
                for row, out in zip(processed, outputs):
                    if not isinstance(row, dict):
                        continue
                    response = getattr(out, "response", None)
                    choices = getattr(response, "choices", None) if response is not None else None
                    if not choices:
                        continue
                    logprobs = getattr(choices[0], "logprobs", None)
                    if logprobs is not None:
                        row["completion_logprobs"] = logprobs
        except Exception as e:
            print(f"[PointwiseJudgeReward] attach completion_logprobs failed: {e}", flush=True)
        return processed

    RolloutTrainerMixin._prepare_rollout_params = _prepare_rollout_params_patched
    RolloutTrainerMixin._postprocess_rollout_outputs = _postprocess_rollout_outputs_patched
    RolloutTrainerMixin._openrubrics_logprob_patch = True
    print("[PointwiseJudgeReward] rollout patch applied (logprobs forwarding)", flush=True)


_patch_rollout_logprobs()


class PointwiseJudgeReward(ORM):
    def __init__(self) -> None:
        super().__init__()

        host = os.environ.get("REWARD_SERVER_HOST", "127.0.0.1")
        port = os.environ.get("REWARD_SERVER_PORT", "8000")
        self.endpoint = f"http://{host}:{port}/score"
        self.timeout = float(os.environ.get("REWARD_TIMEOUT", "300"))

        self._session = requests.Session()
        self._call_count = 0
        self._logged_prob_source = False

        print(f"[PointwiseJudgeReward] endpoint: {self.endpoint}", flush=True)

    @staticmethod
    def _to_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        return [x]

    @staticmethod
    def _weight_from_tags(tags: List[str]) -> float:
        tags = [str(t).strip().lower() for t in (tags or [])]
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
        tokens: List[str] | None = None,
    ) -> Dict[int, Dict[str, float]]:
        context_tail = ""
        waiting_value = False
        active_k: Optional[int] = None
        probs_by_index: Dict[int, Dict[str, float]] = {}

        tokens = tokens or []

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
                return self._extract_probs_from_steps(logprobs, tokens=tokens)

            content = payload.get("content")
            if isinstance(content, list):
                return self._extract_probs_from_openai_style(content)

        if isinstance(payload, list):
            if payload and isinstance(payload[0], dict):
                first = payload[0]
                if "token" in first or "top_logprobs" in first:
                    return self._extract_probs_from_openai_style(payload)
                return self._extract_probs_from_steps(payload, tokens=[])

        return {}

    def _compute_completion_score(
        self,
        completion: str,
        completion_prob_payload: Any,
        rubric_text: str,
    ) -> Tuple[float, bool]:
        rubric_count = count_rubric_items(rubric_text)
        if rubric_count <= 0:
            return 0.0, False

        parsed = parse_json_to_dict(completion)
        if not parsed:
            return 0.0, False

        probs_by_index = self._extract_probs_from_completion_payload(completion_prob_payload)
        if not probs_by_index:
            return 0.0, False

        weights = self._weights_from_rubric_text(rubric_text, rubric_count)
        total = 0.0
        prob_covered = 0
        for i in range(1, rubric_count + 1):
            if i in probs_by_index:
                prob_covered += 1
            prob_info = probs_by_index.get(i, {})
            tp = float(prob_info.get("true_prob", 0.0) or 0.0)
            fp = float(prob_info.get("false_prob", 0.0) or 0.0)
            total += (tp - fp) * float(weights[i - 1])

        fmt_ok = format_is_valid(parsed, rubric_count) and (prob_covered == rubric_count)
        return total, fmt_ok

    def __call__(
        self,
        completions: Sequence[str],
        prompt: Sequence[str] | None = None,
        chosen: Sequence[str] | None = None,
        rejected: Sequence[str] | None = None,
        rubrics: Sequence[Any] | None = None,
        tags: Sequence[str] | None = None,
        tag: Sequence[str] | None = None,
        **kwargs,
    ) -> List[float]:
        self._call_count += 1

        chosen = chosen or kwargs.get("chosen") or []
        rejected = rejected or kwargs.get("rejected") or []
        rubrics = rubrics or kwargs.get("rubrics") or []
        tags = tags or tag or kwargs.get("tags") or kwargs.get("tag") or []

        messages = kwargs.get("messages") or []
        original_prompts = kwargs.get("original_prompt") or []
        prompt_ids = kwargs.get("prompt_id") or kwargs.get("prompt_ids") or []

        completion_probs_raw = (
            kwargs.get("completion_logprobs")
            or kwargs.get("completion_probs")
            or kwargs.get("completions_logprobs")
            or kwargs.get("logprobs")
            or []
        )
        completion_prob_source = None
        for candidate in (
            "completion_logprobs",
            "completion_probs",
            "completions_logprobs",
            "logprobs",
            "completion_prob",
            "completion_logprob",
            "completions_prob",
            "completions_logprob",
        ):
            v = kwargs.get(candidate)
            if v is None:
                continue
            ok = False
            try:
                ok = len(v) > 0  # type: ignore[arg-type]
            except Exception:
                ok = True
            if ok:
                completion_probs_raw = v
                completion_prob_source = candidate
                break

        if completion_prob_source is None:
            for k, v in kwargs.items():
                lk = str(k).lower()
                if ("logprob" not in lk and "prob" not in lk) or v is None:
                    continue
                ok = False
                try:
                    ok = len(v) > 0  # type: ignore[arg-type]
                except Exception:
                    ok = True
                if ok:
                    completion_probs_raw = v
                    completion_prob_source = str(k)
                    break

        completion_probs = self._to_list(completion_probs_raw)
        completion_probs_non_null = sum(1 for p in completion_probs if p is not None)

        if not self._logged_prob_source:
            self._logged_prob_source = True
            prob_like_keys = sorted(
                [str(k) for k in kwargs.keys() if ("prob" in str(k).lower() or "logprob" in str(k).lower())]
            )
            print(
                f"[PointwiseJudgeReward] prob source={completion_prob_source or 'NONE'} "
                f"len={len(completion_probs)} non_null={completion_probs_non_null} prob_keys={prob_like_keys}",
                flush=True,
            )

        def _extract_user_prompt(msg_list):
            if isinstance(msg_list, list):
                for m in msg_list:
                    if isinstance(m, dict) and m.get("role") == "user":
                        return m.get("content", "") or ""
            return ""

        all_prompts = [_extract_user_prompt(msg_list) for msg_list in messages]
        original_prompts_list = list(original_prompts) if original_prompts else []

        n_completions = len(completions)
        chosens = list(chosen) if chosen else []
        rejecteds = list(rejected) if rejected else []
        rubrics_list = list(rubrics) if rubrics else []
        tags_list = list(tags) if tags else ["chosen"] * len(all_prompts)

        n_samples = len(all_prompts)
        if n_samples == 0:
            print("[PointwiseJudgeReward] WARNING: no prompts", flush=True)
            return [0.0] * n_completions

        completions_per_sample = n_completions // n_samples if n_samples > 0 else n_completions
        print(
            f"\n[PointwiseJudgeReward] n_completions={n_completions}, n_samples={n_samples}, k={completions_per_sample}",
            flush=True,
        )

        all_rewards: List[float] = [0.0] * n_completions

        if prompt_ids and len(prompt_ids) == n_completions:
            pid_to_indices: Dict[str, List[int]] = {}
            for idx, pid in enumerate(prompt_ids):
                pid_to_indices.setdefault(str(pid), []).append(idx)

            for pid, idxs in pid_to_indices.items():
                first = idxs[0]
                c = chosens[first] if first < len(chosens) else ""
                r = rejecteds[first] if first < len(rejecteds) else ""
                rub = rubrics_list[first] if first < len(rubrics_list) else []
                sample_tag = tags_list[first] if first < len(tags_list) else "chosen"

                if isinstance(rub, str):
                    rub = [rub]
                elif rub is None:
                    rub = []
                else:
                    rub = [str(x) for x in rub]

                sample_completions = [completions[i] for i in idxs]
                sample_probs = [completion_probs[i] if i < len(completion_probs) else None for i in idxs]
                rubric_text = "\n".join([x.strip() for x in rub if str(x).strip()])
                sample_scores: List[float] = []
                sample_fmts: List[bool] = []
                for comp_text, comp_prob in zip(sample_completions, sample_probs):
                    score, fmt_ok = self._compute_completion_score(comp_text, comp_prob, rubric_text)
                    sample_scores.append(float(score))
                    sample_fmts.append(bool(fmt_ok))

                orig_prompt = original_prompts_list[first] if first < len(original_prompts_list) else ""
                if not orig_prompt or not c or not r or not rub:
                    print(
                        f"[PointwiseJudgeReward] invalid data -> zeros (prompt_id={pid}, "
                        f"orig_prompt={bool(orig_prompt)}, c={bool(c)}, r={bool(r)}, rub={bool(rub)})",
                        flush=True,
                    )
                    for i in idxs:
                        all_rewards[i] = 0.0
                    continue

                try:
                    resp = self._session.post(
                        self.endpoint,
                        json={
                            "prompt": orig_prompt,
                            "chosen": c,
                            "rejected": r,
                            "rubrics": rub,
                            "completions": sample_completions,
                            "completion_scores": sample_scores,
                            "completion_fmts": sample_fmts,
                            "tag": sample_tag,
                        },
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    rewards = resp.json().get("rewards", [])
                    if len(rewards) != len(sample_completions):
                        rewards = (rewards + [0.0] * len(sample_completions))[:len(sample_completions)]
                    for i, rwd in zip(idxs, rewards):
                        all_rewards[i] = rwd
                    print(f"[PointwiseJudgeReward] prompt_id={pid} k={len(sample_completions)} rewards={rewards}", flush=True)
                except Exception as e:
                    print(f"    ERROR: {e}", flush=True)
                    for i in idxs:
                        all_rewards[i] = 0.0
        else:
            for sample_idx in range(n_samples):
                c = chosens[sample_idx] if sample_idx < len(chosens) else ""
                r = rejecteds[sample_idx] if sample_idx < len(rejecteds) else ""
                rub = rubrics_list[sample_idx] if sample_idx < len(rubrics_list) else []
                sample_tag = tags_list[sample_idx] if sample_idx < len(tags_list) else "chosen"

                if isinstance(rub, str):
                    rub = [rub]
                elif rub is None:
                    rub = []
                else:
                    rub = [str(x) for x in rub]

                start_idx = sample_idx * completions_per_sample
                end_idx = start_idx + completions_per_sample
                sample_completions = list(completions[start_idx:end_idx])
                sample_probs = [completion_probs[i] if i < len(completion_probs) else None for i in range(start_idx, end_idx)]
                rubric_text = "\n".join([x.strip() for x in rub if str(x).strip()])
                sample_scores: List[float] = []
                sample_fmts: List[bool] = []
                for comp_text, comp_prob in zip(sample_completions, sample_probs):
                    score, fmt_ok = self._compute_completion_score(comp_text, comp_prob, rubric_text)
                    sample_scores.append(float(score))
                    sample_fmts.append(bool(fmt_ok))

                orig_prompt = original_prompts_list[sample_idx] if sample_idx < len(original_prompts_list) else ""
                if not orig_prompt or not c or not r or not rub:
                    print(
                        f"[PointwiseJudgeReward] invalid data -> zeros (orig_prompt={bool(orig_prompt)}, "
                        f"c={bool(c)}, r={bool(r)}, rub={bool(rub)})",
                        flush=True,
                    )
                    for i in range(start_idx, min(end_idx, n_completions)):
                        all_rewards[i] = 0.0
                    continue

                try:
                    resp = self._session.post(
                        self.endpoint,
                        json={
                            "prompt": orig_prompt,
                            "chosen": c,
                            "rejected": r,
                            "rubrics": rub,
                            "completions": sample_completions,
                            "completion_scores": sample_scores,
                            "completion_fmts": sample_fmts,
                            "tag": sample_tag,
                        },
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    rewards = resp.json().get("rewards", [])
                    if len(rewards) != len(sample_completions):
                        rewards = (rewards + [0.0] * len(sample_completions))[:len(sample_completions)]
                    for i, rwd in enumerate(rewards):
                        if start_idx + i < n_completions:
                            all_rewards[start_idx + i] = rwd
                    print(f"[PointwiseJudgeReward] sample={sample_idx} k={len(sample_completions)} rewards={rewards}", flush=True)
                except Exception as e:
                    print(f"    ERROR: {e}", flush=True)
                    for i in range(start_idx, min(end_idx, n_completions)):
                        all_rewards[i] = 0.0

        if len(all_rewards) != n_completions:
            all_rewards = (all_rewards + [0.0] * n_completions)[:n_completions]

        if all_rewards:
            print(f"[PointwiseJudgeReward] rewards: n={len(all_rewards)}", flush=True)

        return all_rewards


orms["pointwise_judge_reward"] = PointwiseJudgeReward
