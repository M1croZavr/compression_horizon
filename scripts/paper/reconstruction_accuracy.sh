
# fig:pca_reconstruction_accuracy
python scripts/visualize_progressive_embeddings.py \
  --dataset_path $PWD/artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes/ \
  --perplexity_max_samples 128 \
  --perplexity_model unsloth/Meta-Llama-3.1-8B
