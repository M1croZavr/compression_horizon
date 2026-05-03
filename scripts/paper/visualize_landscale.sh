set -e # exit on error
set -x # print commands
set -o pipefail # exit on error in pipes

ALL_CHECKPOINTS=(
    "sl_4096_SmolLM2-1.7B_lr_0.1"
    "sl_4096_SmolLM2-1.7B_lr_0.5"
    "sl_4096_SmolLM2-1.7B_lr_1.0"
    "sl_4096_SmolLM2-1.7B_lowdim_32_lowproj"
    "sl_4096_SmolLM2-1.7B_lowdim_64_lowproj"
    "sl_4096_SmolLM2-1.7B_lowdim_256_lowproj"
    "sl_4096_SmolLM2-1.7B_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8"
)

for checkpoint in "${ALL_CHECKPOINTS[@]}"; do
    python scripts/paper/visualize_landscale_2pca.py --sample_id 0 --batch_size 64 --pca4 --mesh_resolution 30 --num-frames 5 --padding 0.5 --dataset_path artifacts/experiments_progressive/${checkpoint}/progressive_prefixes
done


# ALL_CHECKPOINTS=(
#     "sl_4096_Meta-Llama-3.1-8B_lr_0.1"
#     "sl_4096_Meta-Llama-3.1-8B_lr_1.0"
# )

# for checkpoint in "${ALL_CHECKPOINTS[@]}"; do
#     python scripts/paper/visualize_landscale_2pca.py --batch_size 64 --sample_id 0 --mesh_resolution 30 --num-frames 4 --padding 0.25 --dataset_path artifacts/experiments_progressive/${checkpoint}/progressive_prefixes
# done
#     # python scripts/paper/visualize_landscale_2pca.py --pca4 --batch_size 64 --sample_id 0 --mesh_resolution 30 --num-frames 4 --padding 0.25 --dataset_path artifacts/experiments_progressive/${checkpoint}/progressive_prefixes


python scripts/paper/visualize_landscale_2pca.py --batch_size 64 --sample_id 0 --mesh_resolution 60 --num-frames 6 --anchor_indices 50 100 200 400 800 1000 --neighborhood 300 100 50 50 10 5  --padding 0.25 --dataset_path artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes