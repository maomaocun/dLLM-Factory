-----

# dLLM-Factory

**dLLM-Factory** is a comprehensive project focused on Diffusion Large Language Models (dLLMs). It provides a full suite of implementation code for core modules, including Pre-training, Supervised Fine-tuning (SFT), Reinforcement Learning (RL), and Inference.

[](https://www.star-history.com/#maomaocun/dLLM-Factorye&Timeline)

-----

## 📖 Project Introduction

This project is designed to offer researchers and developers an efficient and user-friendly platform for dLLM training and inference. It supports the entire workflow, from data preprocessing and model training to inference and deployment. The project features a clear and organized structure, making it easy for users to undertake secondary development and customization.

## ✨ 主要特性

- **🧠 预训练（Pretrain）：** 从零开始训练基础模型。
  - 支持数据集：`SlimPajama`

***

- **🔧 有监督微调（SFT）：** 将预训练模型适配到特定任务。
  - 支持数据集：`simplescaling-s1K`

***

- **🤖 强化学习（RL）：** 利用反馈进一步优化模型性能。
  - 支持方法：`diff-grpo`

***

- **🚀 推理（Inference）：** 高效运行已训练模型以应用于实际场景。
  - 支持加速：`dLLM-cache`

***

- **📈 Evaluation:** Comprehensive evaluation on various benchmarks.
  | Benchmark | LLaDA Support | Dream Support |
  | :--- | :---: | :---: |
  | **BBH** | ✅ | ✅ |
  | **GPQA** | ✅ | ✅ |
  | **GSM8K** | ✅ | ✅ |
  | **HumanEval** | ✅ | ✅ |
  | **Long Bench** | ✅ | - |
  | **MBPP** | ✅ | ✅ |
  | **Minerva Math**| ✅ | ✅ |
  | **MMLU** | ✅ | ✅ |
  | **MMLU Pro** | ✅ | ✅ |

***

## 📝 TODO

- [ ] Expand dataset support for pretraining and SFT
- [ ] Integrate more RL algorithms and strategies
- [ ] Add more dLLM acceleration methods (e.g., quantization, pruning, etc.)
- [ ] Add more evaluation benchmarks and metrics
- [ ] Enhance user experience for deployment and customization
## 🛠️ Usage

### Pretraining

Execute the following command to start Pretraining:

```sh
cd pretrain
bash run_pretrain.sh
```

### Supervised Fine-tuning (SFT)

Execute the following command to start supervised fine-tuning:

```sh
cd sft
accelerate launch --config_file ./config/accelerate/lora_config.yaml ./sft.py
```

### Reinforcement Learning (RL)

To begin reinforcement learning, run the provided script:

```sh
cd rl
bash examples/script/train_diffu_grpo.sh
```


### Evalution

Get the evaluation results by this command:

```sh
cd evaluation
bash scripts/Dream/run_Dream_bbh_base.sh
```

## 🙏 Acknowledgments

We extend our sincere gratitude to the following projects for their excellent work. The Reinforcement Learning code in this repository is modified based on their contributions:

  - **[d1](https://github.com/dllm-reasoning/d1):** 一个专注于通过强化学习扩展扩散大语言模型推理能力的项目。
  - **[dLLM-cache](https://github.com/maomaocun/dllm-cache):** 一个用于自适应缓存加速dLLM的实现，已集成到本仓库中。
  - **[TinyLlama](https://github.com/jzhang38/TinyLlama)  [SMDM](https://github.com/ML-GSAI/SMDM):** 本项目的预训练代码参考了这些仓库，非常感谢他们的贡献。

## 📖 Citation
```
@misc{yangyicun2025dLLMFactory,
  title={dLLM-Factory: A Comprehensive Platform for Diffusion Large Language Models},
  author={Yang Yicun and Cheng Shuang and Liu Dawei and Bian Yihan and Zhang Yaojie},
  year={2025},
  url = {https://github.com/maomaocun/dllm-Factory}
}
```

## 📧 Contact

For any questions or collaboration inquiries, feel free to reach out at: [yangyicun187@gmail.com](mailto:yangyicun187@gmail.com)

## :star2: Star History
[![Star History Chart](https://api.star-history.com/svg?repos=maomaocun/dLLM-Factory&type=Timeline)](https://www.star-history.com/#maomaocun/dLLM-Factorye&Timeline)
