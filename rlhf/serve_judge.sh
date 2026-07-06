set -e

export VLLM_NO_USAGE_STATS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

MODEL_PATH="${MODEL_PATH:-OpenRubrics/RubricARROW-8B-Judge}"
HOST=127.0.0.1
PORT=8011

CUDA_VISIBLE_DEVICES=1 \
swift rollout \
  --model_type qwen3 \
  --infer_backend vllm \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.90 \
  --vllm_engine_kwargs '{"load_format":"auto"}' \
  --response_prefix '<think>\n\n</think>\n\n' \
  --host $HOST \
  --port $PORT \
  --model $MODEL_PATH
