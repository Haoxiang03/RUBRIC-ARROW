# RUBRIC-ARROW

Official repository for **RUBRIC-ARROW: Alternating Pointwise Rubric Reward Modeling for LLM Post-training in Non-verifiable Domains**.

[[Paper]](https://arxiv.org/abs/2605.29156)[[Model]](https://huggingface.co/OpenRubrics/RubricARROW)

## Overview

RUBRIC-ARROW is a rubric-based reward modeling framework for evaluating and improving large language models in open-ended, non-verifiable tasks.

The repository includes data for three components of RUBRIC-ARROW:

- **Rubric generator training**: data for learning to generate evaluation criteria.
- **Rubric-conditioned judge training**: data for learning to score responses based on generated rubrics.
- **RLHF training**: preference/post-training data for optimizing LLMs with the rubric-based reward model.

## Environment

This repository is built on the `ms-swift` training framework.

- `ms-swift = 3.11`
- Official repository: https://github.com/modelscope/ms-swift

Please install `ms-swift` following the official instructions before running training scripts.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{jiang2026rubricarrow,
  title={RUBRIC-ARROW: Alternating Pointwise Rubric Reward Modeling for LLM Post-training in Non-verifiable Domains},
  author={Jiang, Haoxiang and Dong, Zihan and Liu, Tianci and Wang, Wanying and Xu, Ran and Yu, Tony and Zhang, Linjun and Wang, Haoyu},
  journal={arXiv preprint arXiv:2605.29156},
  year={2026}
}
