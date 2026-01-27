
# All models progressive training
# tab:attn_hijacking
PYTHONPATH=./src:. python scripts/paper/attn_hijacking.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_SmolLM2-135M_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-360M_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Llama-3.2-1B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Llama-3.2-3B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-160m_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-410m_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-270m_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-1b-pt_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lr_0.1/progressive_prefixes \
  --compute \
  --tablefmt latex


# base vs aligned and low dim projected checkpoints
PYTHONPATH=./src:. python scripts/paper/attn_hijacking.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
  --compute \
  --tablefmt latex


# Prefix Tuning
PYTHONPATH=./src:. python scripts/paper/attn_hijacking.py \
  --checkpoints \
    artifacts/experiments_prefix_tuning/pt_sl_16384_Llama-3.2-3B/progressive_prefixes \
    artifacts/experiments_prefix_tuning/pt_sl_16384_Qwen3-4B/progressive_prefixes \
    artifacts/experiments_prefix_tuning/pt_sl_16384_SmolLM2-1.7B/progressive_prefixes \
    artifacts/experiments_prefix_tuning/pt_sl_16384_SmolLM2-135M/progressive_prefixes \
    artifacts/experiments_prefix_tuning/pt_sl_16384_SmolLM2-360M/progressive_prefixes \
  --compute \
  --tablefmt latex





