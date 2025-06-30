accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks humaneval --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,prompt_interval_steps=-1,gen_interval_steps=-1,transfer_ratio=0,cache_order=0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=32,gen_length=512,steps=512,cfg_scale=0.0 "  \
--output_path ./humaneval_log/ \
--log_samples \
--confirm_run_unsafe_code \

accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks humaneval --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,prompt_interval_steps=50,gen_interval_steps=8,transfer_ratio=0.25,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=32,gen_length=512,steps=512,cfg_scale=0.0 "  \
--output_path ./humaneval_log/ \
--log_samples \
--confirm_run_unsafe_code \


accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks humaneval --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,prompt_interval_steps=25,gen_interval_steps=5,transfer_ratio=0.25,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=32,gen_length=512,steps=512,cfg_scale=0.0 "  \
--output_path ./humaneval_log/ \
--log_samples \
--confirm_run_unsafe_code \