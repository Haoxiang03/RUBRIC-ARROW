# Rubric Generator GRPO

GRPO training code for generating universal response rubrics.

## Data

Three training iterations are included:

- `../datasets/rubric-generator/iter1.jsonl`
- `../datasets/rubric-generator/iter2.jsonl`
- `../datasets/rubric-generator/iter3.jsonl`

Records use this schema:

```json
{
  "messages": [{"role": "user", "content": "rubric generation prompt"}],
  "original_prompt": "request answered by the preference pair",
  "chosen": "preferred response",
  "rejected": "non-preferred response"
}
```

## Run

Start the rubric generator policy:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/path/to/generator \
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

Start training on a separate GPU allocation:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
NPROC_PER_NODE=6 \
MODEL_PATH=/path/to/generator \
OUTPUT_DIR=/path/to/output \
./train.sh
```
