#!/bin/bash
# This script is used to run the pretraining of the Diffusion LLaMA model.
wandb online
export WANDB_API_KEY=xxx

TRAIN_DATASET=(
    "1.0 finewebedu-sample-10BT /dataset_dir/train_fineweb_sample_10TB"
    "1.0 SlimPajama-627B /dataset_dir/train_slim_pajama_627B"
)
DATA_LIST=$(IFS=,; echo "${TRAIN_DATASET[*]}")

# https://lightning.ai/docs/fabric/stable/fundamentals/launch.html
# deepspeed_stage_3
# Mask token ID for the forward process, equal to vocab size
lightning run model \
    --node-rank=$RANK  \
    --main-address=$MASTER_ADDR \
    --devices=${GPUS_PER_NODE} \
    --num-nodes=${WORLD_SIZE} \
    --main-port=$MASTER_PORT \
    --strategy=fsdp \
    --accelerator=cuda \
    --precision="bf16-mixed" \
    pretrain/train_mdm.py \
    --model.name "Diff_LLaMA_336M" \
    --model.mask_token_id 128512 \
    --optimizer.lr 3e-4 \
    --optimizer.weight_decay 0.1 \
    --optimizer.beta1 0.9 \
    --optimizer.beta2 0.95 \
    --lr_scheduler.warmup_steps 1024 \
    --lr_scheduler.decay_steps 20480 \
    --lr_scheduler.decay_type cosine \
    --lr_scheduler.lr_min 0.1 \
    --training.seed 42 \
    --training.output_folder ./output \
    --training.job_name "Diff_LLaMA_336M" \
    --training.log_iter_interval 1 \
    --training.eval_step_interval 999999999 \
    --training.save_step_interval 20 \
    --training.save_total_limit 10 \
    --training.wandb_project "diffusion-research" \
    --training.wandb_name "mdlm" \
    --training.wandb_dir "./wandb" \
    --training.micro_batch_size 4 \
    --training.seq_len 2048 \
    --training.train_data_cfg "${DATA_LIST}" \
    --training.resume True\
    --training.gradient_accumulation_steps 1 \
    --training.max_steps 20480 \
    --training.max_norm 1.0
