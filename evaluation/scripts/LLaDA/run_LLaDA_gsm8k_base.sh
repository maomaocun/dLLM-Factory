accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,prompt_interval_steps=-1,gen_interval_steps=-1,transfer_ratio=0.0,cache_order=0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0 "  \
--num_fewshot 4  \
--output_path ./gsm8k_log \
--log_samples \

accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks gsm8k --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,prompt_interval_steps=100,gen_interval_steps=6,transfer_ratio=0.25,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0 "  \
--num_fewshot 4  \
--output_path ./gsm8k_log \
--log_samples \

accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks gsm8k --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,prompt_interval_steps=25,gen_interval_steps=5,transfer_ratio=0.25,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0 "  \
--num_fewshot 4  \
--output_path ./gsm8k_log \
--log_samples \
