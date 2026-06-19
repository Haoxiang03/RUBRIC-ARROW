# Judge GRPO

GRPO training code for a rubric-based pointwise judge.

## Data

Three training iterations are included:

- `../datasets/judge/iter1.jsonl`
- `../datasets/judge/iter2.jsonl`
- `../datasets/judge/iter3.jsonl`

Records contain these fields:

```json
{
  "messages": [{"role": "user", "content": "judge prompt"}],
  "original_prompt": "the prompt answered by the chosen and rejected responses",
  "chosen": "preferred response",
  "rejected": "non-preferred response",
  "rubrics": ["1. ... [Hard Rule]\n2. ... [Principle]"],
  "tag": "chosen"
}
```

## Run

Start the policy rollout server on its own GPU allocation:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/path/to/model \
./scripts/serve_policy.sh
```

Start the CPU reward API:

```bash
./scripts/serve_reward.sh
```

Start training on a separate GPU allocation:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
NPROC_PER_NODE=6 \
MODEL_PATH=/path/to/model \
OUTPUT_DIR=/path/to/output \
./scripts/train.sh
```