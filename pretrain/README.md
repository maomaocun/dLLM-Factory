# Diffusion LM Pretraining

## Dependency
We can build the Anaconda environment based on [TinyLlama](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md). First install the [TinyLlama](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md) Anaconda environment and then run
```sh
pip install lm-eval==0.4.4 numpy==1.25.0 bitsandbytes==0.43.1
pip install openai==0.28 fschat==0.2.34 anthropic
```
In addition, we provide the conda installation commands in the [CONDA.md](CONDA.md) file for reference and completeness.


## Pretrain
Please first use the code provided by [TinyLlama](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md) to preprocess the 
[SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) dataset and the put the data chunks into `/dataset/slim_star_combined`.


### Pretrain ARMs
```sh
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
```

## Convert to hf format
After pretraining, you can convert the model to Hugging Face format using the following command:

...

## Acknowledgments
This code is based on the [TinyLlama](https://github.com/jzhang38/TinyLlama) repository and [SMDM](https://github.com/ML-GSAI/SMDM) repository. We thank the authors for their contributions and support.
