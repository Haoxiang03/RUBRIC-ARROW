# Rubric Generator GRPO

GRPO training code for generating universal response rubrics.

## Data

Our training data comes from [OpenRubrics](https://huggingface.co/OpenRubrics). The processed data is evenly split into three subsets, which are used for the three alternating training iterations respectively. We apply several transformations to convert the original data into our training format.

Each training example follows the format below:

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
