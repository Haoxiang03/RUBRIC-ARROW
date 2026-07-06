# Rubric-Guided GRPO

Training code for optimizing an answer model with rubric-based rewards.

## Data

Training JSONL records use this schema:

```json
{
  "messages": [{"role": "user", "content": "user request"}],
  "rubrics": [
    "1. The response must ... [Hard Rule]\n2. The response should ... [Principle]"
  ]
}
```

## Run

Start the policy rollout server:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/path/to/policy \
./serve_policy.sh
```

Start the independent rubric judge:

```bash
CUDA_VISIBLE_DEVICES=1 \
JUDGE_MODEL_PATH=/path/to/judge \
./serve_judge.sh
```

Start the CPU reward API:

```bash
./serve_reward.sh
```

Start GRPO training on a separate GPU allocation:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
NPROC_PER_NODE=6 \
MODEL_PATH=/path/to/policy \
OUTPUT_DIR=/path/to/output \
./train.sh
```
