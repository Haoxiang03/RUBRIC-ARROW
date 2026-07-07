# RUBRIC-ARROW

Official repository for **RUBRIC-ARROW: Alternating Pointwise Rubric Reward Modeling for LLM Post-training in Non-verifiable Domains**.

[[Paper]](https://arxiv.org/abs/2605.29156) [[Model]](https://huggingface.co/OpenRubrics)

## Overview

RUBRIC-ARROW is a rubric-based reward modeling framework for evaluating and improving large language models in open-ended, non-verifiable tasks.

The repository includes data and code for three components of RUBRIC-ARROW:

- **Rubric-conditioned judge training**: codebase for learning to score responses based on generated rubrics.
- **Rubric generator training**: codebase for learning to generate evaluation criteria.
- **RLHF training**: codebase for optimizing LLMs with the rubric-based reward model.

## Environment

This repository is built on the `ms-swift` training framework.

- `ms-swift = 3.11`
- Official repository: https://github.com/modelscope/ms-swift

Please install `ms-swift` following the official instructions before running training scripts.

## Alternating Training

The alternating training loop consists of the following steps:

1. Train or SFT warm up a rubric-conditioned pairwise judge.
2. Train the rubric generator, where the judge serves as the external reward model.
3. Construct judge-training prompts with the latest rubrics and preference labels.
4. Train the judge model.
5. Repeat with the newly updated checkpoints.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{jiang2026rubricarrow,
  title={RUBRIC-ARROW: Alternating Pointwise Rubric Reward Modeling for LLM Post-training in Non-verifiable Domains},
  author={Jiang, Haoxiang and Dong, Zihan and Liu, Tianci and Wang, Wanying and Xu, Ran and Yu, Tony and Zhang, Linjun and Wang, Haoyu},
  journal={arXiv preprint arXiv:2605.29156},
  year={2026}
}
