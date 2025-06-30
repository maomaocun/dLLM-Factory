accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks mmlu_pro --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,prompt_interval_steps=-1,gen_interval_steps=-1,cache_order=0,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./mmlu_pro_log \
--log_samples \

accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks mmlu_pro --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,prompt_interval_steps=100,gen_interval_steps=6,cache_order=0,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./mmlu_pro_log \
--log_samples \

