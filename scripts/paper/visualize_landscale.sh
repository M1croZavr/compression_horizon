
set -x

python scripts/paper/visualize_landscale_2pca.py --sample_id 0 --num-frames 10 --padding 0.5 --dataset_path artifacts/experiments_progressive/sl_4096_pythia-1.4b/progressive_prefixes
python scripts/paper/visualize_landscale_2pca.py --sample_id 0 --num-frames 10 --padding 0.5 --dataset_path artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.1/progressive_prefixes
python scripts/paper/visualize_landscale_2pca.py --sample_id 0 --num-frames 10 --padding 0.5 --dataset_path artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes