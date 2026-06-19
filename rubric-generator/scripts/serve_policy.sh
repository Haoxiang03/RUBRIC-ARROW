#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the rubric generator model.}"

exec swift rollout \
  --model "${MODEL_PATH}" \
  --model_type "${MODEL_TYPE:-qwen3}" \
  --use_hf "${USE_HF:-true}" \
  --infer_backend vllm \
  --vllm_tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-1}" \
  --vllm_gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --template "${TEMPLATE:-qwen3}" \
  --template_backend swift \
  --use_chat_template false \
  --response_prefix $'<think>\n\n</think>\n\n' \
  --host "${POLICY_HOST:-127.0.0.1}" \
  --port "${POLICY_PORT:-8010}"
