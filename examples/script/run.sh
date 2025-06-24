#!/bin/bash
export LOGDIR=/cpfs02/shared/llmit6/liudawei/xpuyu_work_dirs/grpo_checkpoints/d1
export HF_HOME=/cpfs02/shared/llmit6/liudawei/hf_cache  # dataset cache
ROOT=$(git rev-parse --show-toplevel)

source /cpfs02/user/chengshuang/anaconda3/etc/profile.d/conda.sh
conda activate /cpfs02/shared/llmit6/liudawei/env/d1
transformers_version=$(python -c "import transformers; print(transformers.__version__)")
echo "Your transformers version: $transformers_version, and we need 4.49.0"

cd $ROOT
DATASET="gsm8k"
MODEL_PATH=/cpfs02/shared/llmit6/liudawei/models/LLaDA-8B-Instruct
RUN_NAME=LLaDA_${DATASET}
mkdir -p $LOGDIR/$RUN_NAME

# copy the config files
# cp $ROOT/examples/configs/diffu_RL/accelerate.yaml $LOGDIR/$RUN_NAME/accelerate.yaml
# cp $ROOT/examples/configs/diffu_RL/train.yaml $LOGDIR/$RUN_NAME/train.yaml
# SCRIPT_NAME=$(basename "$0")
# cp "$0" "$LOGDIR/$RUN_NAME/$SCRIPT_NAME"

export WANDB_MODE=disabled

# If using lora, `load_in_4bit` should be `true` in `train.yaml`
# If using full fine-tuning, `load_in_4bit` should be `false` in `train.yaml`
accelerate launch \
    --config_file examples/configs/diffu_RL/accelerate.yaml \
    --main_process_port 12346 diffu_RL/diffu-grpo/diffu_grpo_train.py \
    --config examples/configs/diffu_RL/train.yaml \
    --model_path $MODEL_PATH \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir $LOGDIR/$RUN_NAME \
    --trust_remote_code
