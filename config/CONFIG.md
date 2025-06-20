# Config Directory Overview

This directory contains configuration files for different training and fine-tuning strategies in the dLLM-Factory project. Each subfolder is dedicated to a specific training method or framework, and contains YAML files that define hyperparameters, model settings, and distributed training options.

## Structure

- `sft/`  
  Contains configuration files for Supervised Fine-Tuning (SFT) tasks.
  - `default_config.yaml`: Specifies the base model, batch size, sequence length, number of epochs, learning rate, gradient accumulation steps, output directory, job name, and training data path. Example fields:
    - `model_name`: Name of the pretrained model to use.
    - `local_batch_size`: Batch size per device.
    - `max_length`: Maximum sequence length for tokenization.
    - `num_epochs`: Number of training epochs.
    - `learning_rate`: Optimizer learning rate.
    - `output_dir`: Directory to save model checkpoints and logs.
    - `train_data`: Path to the training dataset.

- `accelerate/`  
  Contains configuration files for distributed training using the HuggingFace Accelerate and DeepSpeed frameworks.
  - `lora_config.yaml`: Configuration for LoRA (Low-Rank Adaptation) training with DeepSpeed. Includes settings for compute environment, DeepSpeed offloading, ZeRO optimization, distributed type, mixed precision, number of processes, and more.
  - `full_param_config.yaml`: A more comprehensive DeepSpeed configuration, including optimizer parameters, bf16 mixed precision, ZeRO optimization details, logging frequency, and distributed training options.

- `lora/`  
  Contains LoRA-specific configuration files for parameter-efficient fine-tuning.
  - `default_config.yaml`: Defines LoRA rank, alpha, target modules for injection (e.g., `q_proj`, `k_proj`, `v_proj`), dropout rate, bias training, and task type (e.g., `CAUSAL_LM`).

## Usage

- Modify the YAML files in each subfolder to adjust training hyperparameters and distributed training settings according to your hardware and task requirements.
- These configuration files are referenced by training scripts in the main project to initialize models, optimizers, and distributed environments.

## Notes
- Comments in the YAML files provide further explanations for each field.
- For more details on each training strategy, refer to the main project documentation.
