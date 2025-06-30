# dLLM-Factory

dLLM-Factory 是一个关于 dLLM（Diffusion Large Language Model）的完整项目，涵盖了预训练（Pretrain）、监督微调（SFT）、强化学习（RL）以及推理（Inference）等核心模块的实现代码。

## 项目简介

本项目旨在为研究者和开发者提供一个高效、易用的 dLLM 训练与推理平台，支持从数据预处理、模型训练到推理部署的全流程。项目结构清晰，便于二次开发和定制。

## 主要功能

- **预训练（Pretrain）**
- **监督微调（SFT）**
- **强化学习（RL）**
- **推理（Inference）**

```sh
accelerate launch --config_file ./config/accelerate/lora_config.yaml  ./sft_script/sft.py
```

## Reinforcement Learning

```sh
cd dLLM-Factory
bash examples/script/train_diffu_grpo.sh
```

> **Acknowledgments**: We thank [d1](https://github.com/dllm-reasoning/d1),[dLLM-cache](https://github.com/maomaocun/dllm-cache)for their excellent work. RL Codes are modified based on theirs.