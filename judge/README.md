# Judge GRPO

GRPO training code for a rubric-based pointwise judge.

## Data

Our training data comes from [OpenRubrics](https://huggingface.co/OpenRubrics). The processed data is evenly split into three subsets, which are used for the three alternating training iterations respectively. We apply several transformations to convert the original data into our training format.

Each training example follows the format below:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "judge prompt"
    }
  ],
  "original_prompt": "request answered by the preference pair",
  "chosen": "preferred response",
  "rejected": "non-preferred response",
  "rubrics": [
    "1. ... [Hard Rule]\n2. ... [Principle]"
  ],
  "tag": "chosen or rejected"
}
```

## Run

Start the policy rollout server on its own GPU allocation:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/path/to/model \
./serve_policy.sh
```

Start the CPU reward API:

```bash
./serve_reward.sh
```

Start training on a separate GPU allocation:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
NPROC_PER_NODE=6 \
MODEL_PATH=/path/to/model \
OUTPUT_DIR=/path/to/output \
./train.sh
```
