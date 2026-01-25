python scripts/paper/visualize_trajectories.py \
  --checkpoints \
  artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
  artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B/progressive_prefixes \
  --output artifacts/paper/Llama3.1-8B.pdf \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt plain
