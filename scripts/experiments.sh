
# LR check
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.1 --max_sequence_lengths 64 128 256 512
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.5 --max_sequence_lengths 64 128 256 512
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 1.0 --max_sequence_lengths 64 128 256 512
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 5.0 --max_sequence_lengths 64 128 256 512

# Low Dim LR=0.01
# TODO все, что ниже - не запускалось
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 32
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 64
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 128
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 256
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 512

# Low Dim = 512 LR check
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.1 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 512
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.5 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 512
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 1.0 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 512
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 5.0 --max_sequence_lengths 64 128 256 512 --low_dim_projection --low_dim_size 512

# Hybrid alpha
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --hybrid_alpha 1.0 --num_alignment_layers_hybrid 4
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --hybrid_alpha 1.0 --num_alignment_layers_hybrid 8
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --hybrid_alpha 1.0 --num_alignment_layers_hybrid 16
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --hybrid_alpha 1.0 --num_alignment_layers_hybrid 20

# Hybrid alpha + LowProj
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --hybrid_alpha 1.0 --num_alignment_layers_hybrid 8 --low_dim_projection --low_dim_size 32
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --hybrid_alpha 1.0 --num_alignment_layers_hybrid 8 --low_dim_projection --low_dim_size 64
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --hybrid_alpha 1.0 --num_alignment_layers_hybrid 8 --low_dim_projection --low_dim_size 128
python scripts/jobs/run_jobs.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths 64 128 256 512 --hybrid_alpha 1.0 --num_alignment_layers_hybrid 8 --low_dim_projection --low_dim_size 256

