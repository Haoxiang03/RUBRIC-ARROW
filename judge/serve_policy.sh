set -e

export VLLM_NO_USAGE_STATS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export VLLM_HOST_IP=127.0.0.1

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
HOST="127.0.0.1"
PORT=8010

CUDA_VISIBLE_DEVICES=0 \
swift rollout \
  --model_type qwen3 \
  --infer_backend vllm \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.9 \
  --response_prefix '<think>\n\n</think>\n\n' \
  --host $HOST \
  --port $PORT \
  --model $MODEL_PATH
