accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks gsm8k --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,prompt_interval_steps=-1,gen_interval_steps=-1,transfer_ratio=0,cache_order=0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
--num_fewshot 4  \
--output_path ./gsm8k_log \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \

accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks gsm8k --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,prompt_interval_steps=50,gen_interval_steps=7,transfer_ratio=0.25,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
--num_fewshot 4  \
--output_path ./gsm8k_log \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \
