
# dLLM-Factory

**dLLM-Factory** ，**由上海交通大学和上海人工智能实验室开发**，是一个专注于扩散大语言模型（dLLMs）的全面项目。它提供了从预训练、监督微调（SFT）、强化学习（RL）到推理等核心模块的完整实现代码。

---

## 📖 项目介绍

本项目旨在为研究人员和开发者提供一个高效且易用的平台，用于 dLLM 的训练和推理。它涵盖了从数据预处理、模型训练到推理和部署的完整工作流，拥有清晰的结构设计，便于用户进行二次开发和定制。目前已支持 `Dream` 和 `LLaDA`。

## ✨ 主要特性

- **🧠 预训练：** 从零开始训练基础模型。
  - 支持数据集：`SlimPajama`

***

- **🔧 监督微调（SFT）：** 将预训练模型适配到特定任务。
  - 支持数据集：`simplescaling-s1K`

***

- **🤖 强化学习（RL）：** 利用反馈进一步优化模型性能。
  - 支持方法：`diff-grpo`

***

- **🚀 推理：** 高效运行已训练模型以应用于实际场景。
  - 支持加速：`dLLM-cache`

***

- **📈 评估：** 在多种基准测试中进行全面评估。
  | 基准测试       | LLaDA 支持 | Dream 支持 |
  |----------------|------------|------------|
  | **BBH**        | ✅         | ✅         |
  | **GPQA**       | ✅         | ✅         |
  | **GSM8K**      | ✅         | ✅         |
  | **HumanEval**  | ✅         | ✅         |
  | **Long Bench** | ✅         | -          |
  | **MBPP**       | ✅         | ✅         |
  | **Minerva Math** | ✅       | ✅         |
  | **MMLU**       | ✅         | ✅         |
  | **MMLU Pro**   | ✅         | ✅         |

***

## 📝 待办事项

- [ ] 扩展预训练和 SFT 的数据集支持
- [ ] 集成更多 RL 算法和策略
- [ ] 添加更多 dLLM 加速方法（例如量化、剪枝等）
- [ ] 增加更多评估基准和指标
- [ ] 提升部署和定制的用户体验

## 🛠️ 使用方法

### 预训练

执行以下命令开始预训练：

```sh
cd pretrain
bash run_pretrain.sh
```

### 监督微调（SFT）

执行以下命令开始监督微调：

```sh
cd sft
accelerate launch --config_file ./config/accelerate/lora_config.yaml ./sft.py
```

### 强化学习（RL）

运行以下脚本开始强化学习：

```sh
cd rl
bash examples/script/train_diffu_grpo.sh
```

### 评估

通过以下命令获取评估结果：

```sh
cd evaluation
bash scripts/Dream/run_Dream_bbh_base.sh
```

## 🙏 致谢

我们衷心感谢以下项目及其卓越贡献。本仓库中的强化学习代码改编自他们的工作：

- **[d1](https://github.com/dllm-reasoning/d1):** 一个致力于通过强化学习增强 dLLM 推理能力的项目。
- **[dLLM-cache](https://github.com/maomaocun/dllm-cache):** 一个用于自适应缓存加速 dLLM 的实现，已集成到本仓库中。
- **[TinyLlama](https://github.com/jzhang38/TinyLlama) [SMDM](https://github.com/ML-GSAI/SMDM):** 本项目的预训练代码参考了这些仓库，感谢它们的贡献。

## 📖 引用

```
@misc{yangyicun2025dLLMFactory,
  title={dLLM-Factory: A Comprehensive Platform for Diffusion Large Language Models},
  author={Yang Yicun and Cheng Shuang and Liu Dawei and Bian Yihan and Zhang Yaojie, qibiqing,zhanglinfeng},
  year={2025},
  url = {https://github.com/maomaocun/dllm-Factory}
}
```

## 📧 联系方式

如有任何问题或合作意向，请随时通过以下邮箱联系： [yangyicun187@gmail.com](mailto:yangyicun187@gmail.com)

## :star2: 星标历史
[![Star History Chart](https://api.star-history.com/svg?repos=maomaocun/dLLM-Factory&type=Timeline)](https://www.star-history.com/#maomaocun/dLLM-Factorye&Timeline)

