# compression_horizon

## TODOs

### Embeddings initialization ablation

### Compare activations diff with different prefix tokens for the same sequence and plot its PCA progressively increasing sequence length

## Linear probing for different NLP Tasks


### Add script for github actions exeriment run.


### Save artifacs. Add telegram for for artifacs exploration.

uv run python ./scripts/reproduction.py --model_checkpoint unsloth/Llama-3.2-1B --limit_dataset_items 10 --max_sequence_length 512 --number_of_mem_tokens 1 --max_optimization_steps_per_sample 5000 --random_seed 1337 --per_device_train_batch_size 10 --learning_rate 0.01 --adam_beta1 0.9 --adam_beta2 0.9 --weight_decay 0.01 --dtype bfloat16

uv run python ./scripts/reproduction.py --model_checkpoint unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --max_sequence_length 1568 --number_of_mem_tokens 1 --max_optimization_steps_per_sample 5000 --random_seed 1337 --per_device_train_batch_size 2 --learning_rate 0.01 --adam_beta1 0.9 --adam_beta2 0.9 --weight_decay 0.01 --dtype bfloat16

debug
uv run python ./scripts/reproduction.py --model_checkpoint unsloth/Llama-3.2-1B --limit_dataset_items 2 --max_sequence_length 128 --number_of_mem_tokens 1 --max_optimization_steps_per_sample 5000 --random_seed 1337 --per_device_train_batch_size 2 --learning_rate 0.01 --adam_beta1 0.9 --adam_beta2 0.9 --weight_decay 0.01 --dtype bfloat16