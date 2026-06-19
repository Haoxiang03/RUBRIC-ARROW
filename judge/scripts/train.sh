#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a base model or checkpoint.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints and logs.}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${ROOT_DIR}/../datasets/judge"
ITERATION="${ITERATION:-1}"
TRAIN_DATASET="${TRAIN_DATASET:-${DATASET_DIR}/iter${ITERATION}.jsonl}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-8010}"
REWARD_SERVER_HOST="${REWARD_SERVER_HOST:-127.0.0.1}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-8000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

args=(
  --rlhf_type grpo
  --model "${MODEL_PATH}"
  --model_type "${MODEL_TYPE:-qwen3}"
  --train_type full
  --use_vllm true
  --vllm_mode server
  --vllm_server_host "${POLICY_HOST}"
  --vllm_server_port "${POLICY_PORT}"
  --dataset "${TRAIN_DATASET}"
  --output_dir "${OUTPUT_DIR}"
  --external_plugins "${ROOT_DIR}/reward_plugin.py"
  --reward_funcs pairwise_judge_reward
  --reward_weights 1.0
  --num_generations "${NUM_GENERATIONS:-6}"
  --max_completion_length "${MAX_COMPLETION_LENGTH:-1024}"
  --per_device_train_batch_size "${BATCH_SIZE:-8}"
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-4}"
  --learning_rate "${LEARNING_RATE:-1e-6}"
  --num_train_epochs "${EPOCHS:-1}"
  --deepspeed "${DEEPSPEED:-zero3}"
  --torch_dtype "${TORCH_DTYPE:-bfloat16}"
  --temperature "${TEMPERATURE:-1.0}"
  --top_p "${TOP_P:-0.95}"
  --num_iterations "${NUM_ITERATIONS:-2}"
  --epsilon_high "${EPSILON_HIGH:-0.28}"
  --beta "${BETA:-0.001}"
  --lr_scheduler_type constant
  --save_steps "${SAVE_STEPS:-120}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-1}"
  --logging_steps "${LOGGING_STEPS:-1}"
  --log_completions false
  --report_to "${REPORT_TO:-none}"
)

NPROC_PER_NODE="${NPROC_PER_NODE}" swift rlhf "${args[@]}"