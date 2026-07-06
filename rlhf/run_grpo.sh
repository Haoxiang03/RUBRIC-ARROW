set -e

export REWARD_SERVER_HOST=127.0.0.1
export REWARD_SERVER_PORT=8000
export REWARD_TIMEOUT=300

export VLLM_HOST_IP=127.0.0.1
export VLLM_SERVER_PORT=8010

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/datasets}"

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_SHM_DISABLE=0

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/rlhf/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/rlhf/qwen3_4b_instruct_rlhf}"
PLUGIN_PATH="${PLUGIN_PATH:-$SCRIPT_DIR/reward_plugin.py}"

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
NPROC_PER_NODE=6 \
swift rlhf \
  --rlhf_type grpo \
  --train_type full \
  --vllm_mode server \
  --vllm_server_host $VLLM_HOST_IP \
  --vllm_server_port $VLLM_SERVER_PORT \
  --output_dir $OUTPUT_DIR \
  --use_vllm true \
  --dataset $DATASET_PATH \
  --reward_funcs rubric_answer_reward \
  --reward_weights 1.0 \
  --external_plugins $PLUGIN_PATH \
  --num_generations 6 \
  --max_completion_length 2024 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-7 \
  --deepspeed zero3 \
  --temperature 1.0 \
  --top_p 0.95 \
  --logging_steps 1 \
  --torch_dtype bfloat16 \
  --log_completions true \
  --log_entropy false \
  --num_train_epochs 1 \
  --lr_scheduler_type constant \
  --epsilon_high 0.28 \
  --num_iterations 2 \
  --beta 0.001 \
  --use_liger_kernel true \
  --model $RESOLVED_MODEL_PATH 