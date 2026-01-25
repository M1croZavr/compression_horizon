# python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 1 --learning_rate 0.01 --max_seq_len 256 --dataset_name mrsndmn/pg19
# python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 1 --learning_rate 0.01 --max_seq_len 256 --dataset_name mrsndmn/pg19-random-suffix-shuffle-64
# python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 1 --learning_rate 0.01 --max_seq_len 256 --dataset_name mrsndmn/pg19-model-sampled-prefix-64-max_len-2048
# python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 1 --learning_rate 0.01 --max_seq_len 256 --dataset_name mrsndmn/pg19-lowercased-partial-64
# python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 1 --learning_rate 0.1 --max_seq_len 256 --dataset_name mrsndmn/pg19
# python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 1 --learning_rate 0.1 --max_seq_len 256 --dataset_name mrsndmn/pg19-random-suffix-shuffle-64
# python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 1 --learning_rate 0.1 --max_seq_len 256 --dataset_name mrsndmn/pg19-model-sampled-prefix-64-max_len-2048
# python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 1 --learning_rate 0.1 --max_seq_len 256 --dataset_name mrsndmn/pg19-lowercased-partial-64


set -x

python scripts/paper/visualize_trajectories.py \
  --checkpoints \
    TODO \
    artifacts/experiments_progressive/sl_2048_Meta-Llama-3.1-8B_ds_pg19-lowercased-partial-64/progressive_prefixes \
    artifacts/experiments_progressive/sl_2048_Meta-Llama-3.1-8B_ds_pg19-random-suffix-shuffle-64/progressive_prefixes \
  --output artifacts/paper/Llama3.1-8B-text-modifications.pdf \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt plain
