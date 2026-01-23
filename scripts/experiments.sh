
# LR check
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.1 --max_sequence_lengths 64 128 256 512
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.5 --max_sequence_lengths 64 128 256 512
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 1.0 --max_sequence_lengths 64 128 256 512
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 5.0 --max_sequence_lengths 64 128 256 512

# Low Dim LR=0.001
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 32
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 64
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 128
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 256
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 512

# Low Dim = 512 LR check
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.1 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 512
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.5 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 512
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 1.0 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 512
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 5.0 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 512

# Hybrid alpha
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 4
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 16
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 20

# Hybrid alpha + LowProj
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8 --low_dim_projection --low_dim_size 32
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8 --low_dim_projection --low_dim_size 64
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8 --low_dim_projection --low_dim_size 128
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8 --low_dim_projection --low_dim_size 256

