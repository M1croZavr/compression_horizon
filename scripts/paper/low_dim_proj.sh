
set -x

PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Llama-3.2-1B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Llama-3.2-3B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-160m_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-410m_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-360M_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-135M_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lr_0.1/progressive_prefixes \
  --output /tmp/trajectories_comparison.png \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt latex


PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_2048_Meta-Llama-3.1-8B/progressive_prefixes \
    artifacts/experiments_progressive/sl_2048_Meta-Llama-3.1-8B_ds_pg19-lowercased-partial-64/progressive_prefixes \
    artifacts/experiments_progressive/sl_2048_Meta-Llama-3.1-8B_ds_pg19-lowercased/progressive_prefixes \
    artifacts/experiments_progressive/sl_2048_Meta-Llama-3.1-8B_ds_pg19-random-suffix-shuffle-64/progressive_prefixes \
  --output /tmp/trajectories_comparison.png \
  --names_mapping "base,lowercased-partial,lowercased,random-suffix" \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt latex


PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.01/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_1.0/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_5.0/progressive_prefixes \
  --names_mapping "0.01,0.1,0.5,1.0,5.0" \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt latex


PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_2048_Meta-Llama-3.1-8B/progressive_prefixes \
    artifacts/experiments_progressive/sl_2048_Meta-Llama-3.1-8B_ds_pg19-model-sampled-llama3.1-8B-prefix-64-max_len-2048/progressive_prefixes \
    artifacts/experiments_progressive/sl_2048_Meta-Llama-3.1-8B_ds_pg19-model-sampled-llama3.1-8B-prefix-64-max_len-2048-temperature-1.0/progressive_prefixes \
    artifacts/experiments_progressive/sl_2048_Meta-Llama-3.1-8B_ds_pg19-model-sampled-llama3.1-8B-prefix-64-max_len-2048-temperature-2.0/progressive_prefixes \
  --output /tmp/trajectories_comparison.png \
  --names_mapping "base,model-sampled,model-sampled_t=1.0,model-sampled_t=2.0" \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt latex