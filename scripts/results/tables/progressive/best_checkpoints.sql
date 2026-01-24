
-- pythia
sl_4096_pythia-160m_lr_0.1
sl_4096_pythia-410m_lr_0.1
sl_4096_pythia-1.4b_lr_1.0

-- smollm2
sl_4096_SmolLM2-360M_lr_1.0
sl_4096_SmolLM2-135M_lr_0.5
sl_4096_SmolLM2-1.7B_lr_5.0

-- Llamas
sl_4096_Llama-3.2-1B_lr_1.0
sl_4096_Llama-3.2-3B_lr_1.0
sl_4096_Meta-Llama-3.1-8B_lr_5.0


-- Check how single model different LR affects optimization?
PYTHONPATH=./src:. python scripts/paper/pca_vs_sequence_length.py --dataset_path \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_1.0/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_5.0/progressive_prefixes \
    --out_file_suffix "Llama3.1-8B_all_lrs"
    --target_seq_lengths 128, 256, 384, 512, 640, 768, 896, 1024




