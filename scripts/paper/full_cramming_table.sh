
# For full and progressive checkpoints
python scripts/paper/full_cramming_table.py

# For prefix tuned checkpoints tab:prefix_tuning_accuracy
python scripts/paper/full_cramming_table.py --type prefix_tuning


# python scripts/jobs/run_jobs_prefix_tuning.py --embedding_init_method random0.02 --model pythia --learning_rate 0.01 --limit_dataset_items 10 --max_seq_len 1024
# python scripts/jobs/run_jobs_prefix_tuning.py --embedding_init_method random0.02 --model pythia --learning_rate 0.01 --limit_dataset_items 10 --max_seq_len 2048
# python scripts/jobs/run_jobs_prefix_tuning.py --embedding_init_method random0.02 --model pythia --learning_rate 0.01 --limit_dataset_items 10 --max_seq_len 4096
# python scripts/jobs/run_jobs_prefix_tuning.py --embedding_init_method random0.02 --model pythia --learning_rate 0.01 --limit_dataset_items 10 --max_seq_len 8192
# python scripts/jobs/run_jobs_prefix_tuning.py --embedding_init_method random0.02 --model pythia --learning_rate 0.01 --limit_dataset_items 10 --max_seq_len 16384

# python scripts/jobs/run_jobs_prefix_tuning.py --embedding_init_method random0.02 --model Meta-Llama --learning_rate 0.01 --limit_dataset_items 10 --max_seq_len 1024
# python scripts/jobs/run_jobs_prefix_tuning.py --embedding_init_method random0.02 --model Meta-Llama --learning_rate 0.01 --limit_dataset_items 10 --max_seq_len 2048
# python scripts/jobs/run_jobs_prefix_tuning.py --embedding_init_method random0.02 --model Meta-Llama --learning_rate 0.01 --limit_dataset_items 10 --max_seq_len 4096
# python scripts/jobs/run_jobs_prefix_tuning.py --embedding_init_method random0.02 --model Meta-Llama --learning_rate 0.01 --limit_dataset_items 10 --max_seq_len 8192
# python scripts/jobs/run_jobs_prefix_tuning.py --embedding_init_method random0.02 --model Meta-Llama --learning_rate 0.01 --limit_dataset_items 10 --max_seq_len 16384

# python scripts/jobs/run_jobs_prefix_tuning.py --embedding_init_method random0.02 --model Llama-3.2-1B --learning_rate 0.01 --limit_dataset_items 10 --max_seq_len 1024
# python scripts/jobs/run_jobs_prefix_tuning.py --embedding_init_method random0.02 --model Llama-3.2-1B --learning_rate 0.01 --limit_dataset_items 10 --max_seq_len 2048
# python scripts/jobs/run_jobs_prefix_tuning.py --embedding_init_method random0.02 --model Llama-3.2-1B --learning_rate 0.01 --limit_dataset_items 10 --max_seq_len 4096
# python scripts/jobs/run_jobs_prefix_tuning.py --embedding_init_method random0.02 --model Llama-3.2-1B --learning_rate 0.01 --limit_dataset_items 10 --max_seq_len 8192
# python scripts/jobs/run_jobs_prefix_tuning.py --embedding_init_method random0.02 --model Llama-3.2-1B --learning_rate 0.01 --limit_dataset_items 10 --max_seq_len 16384