accelerate launch  --multi_gpu --mixed_precision=fp16 --num_processes=4 run_wav2vec2.py \
	--dataset_name="kojima-lab/bird-jp-all" \
	--dataset_split_names train \
	--dataset_config_names clean \
	--validation_split_percentage 10 \
	--model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
	--output_dir="./wav2vec2-bird-jp-all" \
	--max_train_steps="20000" \
	--num_warmup_steps="32000" \
	--learning_rate="0.001" \
	--weight_decay="0.0001" \
	--max_duration_in_seconds="100.0" \
	--min_duration_in_seconds="0.2" \
	--logging_steps="100" \
	--saving_steps="100000" \
	--per_device_train_batch_size="8" \
	--per_device_eval_batch_size="8" \
	--adam_beta1="0.9" \
	--adam_beta2="0.98" \
	--adam_epsilon="1e-06" \
	--gradient_checkpointing \
