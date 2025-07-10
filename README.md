# dLLM-Factory

**dLLM-Factory** is a robust and comprehensive project centered on Diffusion Large Language Models (dLLMs). It offers a complete suite of implementation code for essential modules, including Pre-training, Supervised Fine-tuning (SFT), Reinforcement Learning (RL), and Inference.

---

## 📖 Project Introduction

This project **developed by SJTU and Shanghai AI Lab** aims to provide researchers and developers with an efficient, user-friendly platform for training and deploying dLLMs. It encompasses the full workflow, from data preprocessing and model training to inference and deployment, featuring a well-organized structure that facilitates secondary development and customization. Support for `Dream` and `LLaDA` is already included.

## ✨ Key Features

- **🧠 Pre-training:** Train foundational models from scratch.
  - Supported datasets: `SlimPajama`

***

- **🔧 Supervised Fine-tuning (SFT):** Adapt pre-trained models to specific tasks.
  - Supported datasets: `simplescaling-s1K`

***

- **🤖 Reinforcement Learning (RL):** Optimize model performance using feedback.
  - Supported methods: `diff-grpo`

***

- **🚀 Inference:** Efficiently run trained models for real-world applications.
  - Supported accelerations: `dLLM-cache`

***

- **📈 Evaluation:** Thorough assessment across diverse benchmarks.
  | Benchmark   | LLaDA Support | Dream Support |
  |-------------|---------------|---------------|
  | **BBH**     | ✅            | ✅            |
  | **GPQA**    | ✅            | ✅            |
  | **GSM8K**   | ✅            | ✅            |
  | **HumanEval**| ✅            | ✅            |
  | **Long Bench** | ✅          | -             |
  | **MBPP**    | ✅            | ✅            |
  | **Minerva Math** | ✅         | ✅            |
  | **MMLU**    | ✅            | ✅            |
  | **MMLU Pro** | ✅            | ✅            |

***

## 📝 TODO

- [ ] Broaden dataset support for pretraining and SFT
- [ ] Incorporate additional RL algorithms and strategies
- [ ] Introduce more dLLM acceleration techniques (e.g., quantization, pruning, etc.)
- [ ] Expand evaluation benchmarks and metrics
- [ ] Improve user experience for deployment and customization

## 🛠️ Usage

### Pretraining

Initiate pretraining with the following command:

```sh
cd pretrain
bash run_pretrain.sh
```

### Supervised Fine-tuning (SFT)

Start supervised fine-tuning with this command:

```sh
cd sft
accelerate launch --config_file ./config/accelerate/lora_config.yaml ./sft.py
```

### Reinforcement Learning (RL)

Launch reinforcement learning using the provided script:

```sh
cd rl
bash examples/script/train_diffu_grpo.sh
```

### Evaluation

Obtain evaluation results with this command:

```sh
cd evaluation
bash scripts/Dream/run_Dream_bbh_base.sh
```

## 🙏 Acknowledgments

We express our heartfelt thanks to the following projects for their outstanding contributions. The Reinforcement Learning code in this repository has been adapted from their work:

- **[d1](https://github.com/dllm-reasoning/d1):** A project dedicated to enhancing dLLM reasoning capabilities through reinforcement learning.
- **[dLLM-cache](https://github.com/maomaocun/dllm-cache):** An implementation for adaptive caching to accelerate dLLMs, now integrated into this repository.
- **[TinyLlama](https://github.com/jzhang38/TinyLlama) [SMDM](https://github.com/ML-GSAI/SMDM):** The pretraining code in this project draws inspiration from these repositories, and we are deeply grateful for their contributions.

## 📖 Citation

```
@misc{yangyicun2025dLLMFactory,
  title={dLLM-Factory: A Comprehensive Platform for Diffusion Large Language Models},
  author={Yang Yicun and Cheng Shuang and Liu Dawei and Bian Yihan and Zhang Yaojie, qibiqing,zhanglinfeng},
  year={2025},
  url = {https://github.com/maomaocun/dllm-Factory}
}
```

## 📧 Contact

For any questions or collaboration inquiries, feel free to reach out at: [yangyicun187@gmail.com](mailto:yangyicun187@gmail.com)

## :star2: Star History
[![Star History Chart](https://api.star-history.com/svg?repos=maomaocun/dLLM-Factory&type=Timeline)](https://www.star-history.com/#maomaocun/dLLM-Factorye&Timeline)

