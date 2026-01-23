
# LR check
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.01
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.1
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.5
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 1.0
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 5.0

# Low Dim LR=0.001
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.01 --low_dim_projection --low_dim_size 32
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.01 --low_dim_projection --low_dim_size 64
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.01 --low_dim_projection --low_dim_size 128
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.01 --low_dim_projection --low_dim_size 256
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.01 --low_dim_projection --low_dim_size 512

# Low Dim = 512 LR check
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.1 --low_dim_projection --low_dim_size 512
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.5 --low_dim_projection --low_dim_size 512
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 1.0 --low_dim_projection --low_dim_size 512
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 5.0 --low_dim_projection --low_dim_size 512

# Hybrid alpha
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.01 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 4
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.01 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.01 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 16
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.01 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 20

# Hybrid alpha + LowProj
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.01 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8 --low_dim_projection --low_dim_size 32
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.01 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8 --low_dim_projection --low_dim_size 64
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.01 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8 --low_dim_projection --low_dim_size 128
python scripts/jobs/run_jobs_progressive.py --model unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --learning_rate 0.01 --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8 --low_dim_projection --low_dim_size 256

