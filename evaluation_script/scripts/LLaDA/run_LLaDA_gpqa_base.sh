accelerate launch --config_file accelerate_config.yaml  evaluation_script.py -m lm_eval --model LLaDA --tasks gpqa_main_generative_n_shot  --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0,remasking="low_confidence" "  \
--num_fewshot 5  \
--output_path ./gpqa_log \
--log_samples \
--confirm_run_unsafe_code \
--trust_remote_code \

accelerate launch --config_file accelerate_config.yaml  evaluation_script.py -m lm_eval --model LLaDA --tasks gpqa_main_generative_n_shot  --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,prompt_interval_steps=100,gen_interval_steps=8,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0,remasking="low_confidence" "  \
--num_fewshot 5  \
--output_path ./gpqa_log \
--log_samples \
--confirm_run_unsafe_code \
--trust_remote_code \