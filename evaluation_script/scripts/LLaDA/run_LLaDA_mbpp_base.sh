accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks mbpp --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,prompt_interval_steps=-1,gen_interval_steps=-1,cache_order=0,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=32,gen_length=256,steps=256,cfg_scale=0.0,remasking="low_confidence" "  \
--num_fewshot 3  \
--output_path ./mbpp_log \
--log_samples \
--confirm_run_unsafe_code \

accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks mbpp --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,prompt_interval_steps=50,gen_interval_steps=4,cache_order=0,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=32,gen_length=256,steps=256,cfg_scale=0.0,remasking="low_confidence" "  \
--num_fewshot 3  \
--output_path ./mbpp_log \
--log_samples \
--confirm_run_unsafe_code \

accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks mbpp --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,prompt_interval_steps=25,gen_interval_steps=4,cache_order=0,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=32,gen_length=256,steps=256,cfg_scale=0.0,remasking="low_confidence" "  \
--num_fewshot 3  \
--output_path ./mbpp_log \
--log_samples \
--confirm_run_unsafe_code \