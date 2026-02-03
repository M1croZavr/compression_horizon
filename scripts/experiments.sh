
set -x


# MODEL_NAME=unsloth/Llama-3.2-1B
# SEQ_LENGTHS=(256 512 768)


# MODEL_NAME=unsloth/Llama-3.2-3B
# SEQ_LENGTHS=(512 768 1024 1152)

# MODEL_NAME=unsloth/Meta-Llama-3.1-8B
# SEQ_LENGTHS=(1280 1568 1793)


# MODEL_NAME=EleutherAI/pythia-160m
# SEQ_LENGTHS=(32 64 96 128)

# MODEL_NAME=EleutherAI/pythia-410m
# SEQ_LENGTHS=(32 64 96 128)

# MODEL_NAME=EleutherAI/pythia-1.4b
# SEQ_LENGTHS=(64 96 128 160 256)

# MODEL_NAME=unsloth/gemma-3-4b-pt
# SEQ_LENGTHS=(32 64 96 128 160 256)


# LR check
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths $SEQ_LENGTHS
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.1 --max_sequence_lengths $SEQ_LENGTHS
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.5 --max_sequence_lengths $SEQ_LENGTHS
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 1.0 --max_sequence_lengths $SEQ_LENGTHS
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 5.0 --max_sequence_lengths $SEQ_LENGTHS

# Low Dim LR=0.01
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths $SEQ_LENGTHS --low_dim_projection --low_dim_size 32
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths $SEQ_LENGTHS --low_dim_projection --low_dim_size 64
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths $SEQ_LENGTHS --low_dim_projection --low_dim_size 128
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths $SEQ_LENGTHS --low_dim_projection --low_dim_size 256
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths $SEQ_LENGTHS --low_dim_projection --low_dim_size 512

# Low Dim = 512 LR check
# python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.1 --max_sequence_lengths $SEQ_LENGTHS --low_dim_projection --low_dim_size 512
# python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.5 --max_sequence_lengths $SEQ_LENGTHS --low_dim_projection --low_dim_size 512
# python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 1.0 --max_sequence_lengths $SEQ_LENGTHS --low_dim_projection --low_dim_size 512
# python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 5.0 --max_sequence_lengths $SEQ_LENGTHS --low_dim_projection --low_dim_size 512

# Hybrid alpha
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths $SEQ_LENGTHS --hybrid_alpha 1.0 --num_alignment_layers_hybrid 4
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths $SEQ_LENGTHS --hybrid_alpha 1.0 --num_alignment_layers_hybrid 8
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths $SEQ_LENGTHS --hybrid_alpha 1.0 --num_alignment_layers_hybrid 16
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths $SEQ_LENGTHS --hybrid_alpha 1.0 --num_alignment_layers_hybrid 20

# Hybrid alpha + LowProj
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths $SEQ_LENGTHS --hybrid_alpha 1.0 --num_alignment_layers_hybrid 8 --low_dim_projection --low_dim_size 32
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths $SEQ_LENGTHS --hybrid_alpha 1.0 --num_alignment_layers_hybrid 8 --low_dim_projection --low_dim_size 64
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths $SEQ_LENGTHS --hybrid_alpha 1.0 --num_alignment_layers_hybrid 8 --low_dim_projection --low_dim_size 128
python scripts/jobs/run_jobs.py --model_checkpoint $MODEL_NAME --limit_dataset_items 10 --per_device_train_batch_size 10 --max_optimization_steps_per_sample 10000 --learning_rate 0.01 --max_sequence_lengths $SEQ_LENGTHS --hybrid_alpha 1.0 --num_alignment_layers_hybrid 8 --low_dim_projection --low_dim_size 256


