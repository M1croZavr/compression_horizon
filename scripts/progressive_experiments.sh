
# MODEL_NAME=EleutherAI/pythia-160m
# MODEL_NAME=EleutherAI/pythia-410m
# MODEL_NAME=EleutherAI/pythia-1.4b

# MODEL_NAME=unsloth/Llama-3.2-1B
# MODEL_NAME=unsloth/Llama-3.2-3B
# MODEL_NAME=unsloth/Meta-Llama-3.1-8B

# MODEL_NAME=HuggingFaceTB/SmolLM2-135M
# MODEL_NAME=HuggingFaceTB/SmolLM2-360M
# MODEL_NAME=HuggingFaceTB/SmolLM2-1.7B

# MODEL_NAME=unsloth/gemma-3-270m
# MODEL_NAME=unsloth/gemma-3-1b-pt
# MODEL_NAME=unsloth/gemma-3-4b-pt

# LR check
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.01
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.1
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.5
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 1.0
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 5.0

# Low Dim LR=0.001
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.01 --low_dim_projection --low_dim_size 32
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.01 --low_dim_projection --low_dim_size 64
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.01 --low_dim_projection --low_dim_size 128
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.01 --low_dim_projection --low_dim_size 256
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.01 --low_dim_projection --low_dim_size 512

# Low Dim = 512 LR check
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.1 --low_dim_projection --low_dim_size 512
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.5 --low_dim_projection --low_dim_size 512
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 1.0 --low_dim_projection --low_dim_size 512
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 5.0 --low_dim_projection --low_dim_size 512

# Hybrid alpha
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.01 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 4
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.01 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.01 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 16
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.01 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 20

# Hybrid alpha + LowProj
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.01 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8 --low_dim_projection --low_dim_size 32
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.01 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8 --low_dim_projection --low_dim_size 64
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.01 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8 --low_dim_projection --low_dim_size 128
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 10 --learning_rate 0.01 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8 --low_dim_projection --low_dim_size 256


# No BOS token
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.1 --no_bos_token
python scripts/jobs/run_jobs_progressive.py --model EleutherAI/pythia-1.4b --limit_dataset_items 10 --learning_rate 0.5 --no_bos_token
python scripts/jobs/run_jobs_progressive.py --model HuggingFaceTB/SmolLM2-1.7B --limit_dataset_items 10 --learning_rate 0.1 --no_bos_token
python scripts/jobs/run_jobs_progressive.py --model unsloth/gemma-3-4b-pt --limit_dataset_items 10 --learning_rate 0.1 --no_bos_token

