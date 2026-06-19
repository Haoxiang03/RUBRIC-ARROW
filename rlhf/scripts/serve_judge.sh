#!/usr/bin/env bash
set -euo pipefail

: "${JUDGE_MODEL_PATH:?Set JUDGE_MODEL_PATH to the rubric judge model.}"

exec swift rollout \
  --model "${JUDGE_MODEL_PATH}" \
  --model_type "${JUDGE_MODEL_TYPE:-qwen3}" \
  --use_hf "${USE_HF:-true}" \
  --load_args false \
  --infer_backend vllm \
  --vllm_tensor_parallel_size "${JUDGE_TENSOR_PARALLEL_SIZE:-1}" \
  --vllm_gpu_memory_utilization "${JUDGE_GPU_MEMORY_UTILIZATION:-0.9}" \
  --template "${JUDGE_TEMPLATE:-qwen3}" \
  --template_backend swift \
  --use_chat_template false \
  --response_prefix $'<think>\n\n</think>\n\n' \
  --host "${JUDGE_HOST:-127.0.0.1}" \
  --port "${JUDGE_PORT:-8011}"
