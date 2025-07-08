-----

# dLLM-Factory

**dLLM-Factory** is a comprehensive project focused on Diffusion Large Language Models (dLLMs). It provides a full suite of implementation code for core modules, including Pre-training, Supervised Fine-tuning (SFT), Reinforcement Learning (RL), and Inference.

[](https://www.star-history.com/#maomaocun/dLLM-Factorye&Timeline)

-----

## 📖 Project Introduction

This project is designed to offer researchers and developers an efficient and user-friendly platform for dLLM training and inference. It supports the entire workflow, from data preprocessing and model training to inference and deployment. The project features a clear and organized structure, making it easy for users to undertake secondary development and customization.

## ✨ Main Features

- **🧠 Pre-training (Pretrain):** Foundation model training from scratch.
  | Category | Supported |
  | :--- | :--- |
  | **Dataset** | `SlimPajama` |

***

- **🔧 Supervised Fine-tuning (SFT):** Adapting pre-trained models to specific tasks.
  | Category | Supported |
  | :--- | :--- |
  | **Dataset** | `simplescaling-s1K` |

***

- **🤖 Reinforcement Learning (RL):** Further optimizing model performance using feedback.
  | Category | Supported |
  | :--- | :--- |
  | **Method** | `diff-grpo` |

***

- **🚀 Inference:** Efficiently running the trained models for real-world applications.
  | Category | Supported |
  | :--- | :--- |
  | **Acceleration**| `dLLM-cache` |

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
accelerate launch --config_file ./config/accelerate/lora_config.yaml ./sft_script/sft.py
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

  - **[d1](https://github.com/dllm-reasoning/d1):** A project focused on scaling reasoning in Diffusion Large Language Models via Reinforcement Learning.
  - **[dLLM-cache](https://github.com/maomaocun/dllm-cache):** An implementation for accelerating dLLMs with adaptive caching, which has been integrated into our repository.

## 📧 Contact

For any questions or collaboration inquiries, feel free to reach out at: [yangyicun187@gmail.com](mailto:yangyicun187@gmail.com)

## :star2: Star History
[![Star History Chart](https://api.star-history.com/svg?repos=maomaocun/dLLM-Factory&type=Timeline)](https://www.star-history.com/#maomaocun/dLLM-Factorye&Timeline)
