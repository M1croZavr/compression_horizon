"""Compression trainers: different setup trainers."""

from compression_horizon.train.compression_head_trainer import CompressionHeadTrainer
from compression_horizon.train.full_cramming_trainer import FullCrammingTrainer
from compression_horizon.train.low_dim_trainer import LowDimTrainer
from compression_horizon.train.prefix_tuning_trainer import PrefixTuningTrainer
from compression_horizon.train.progressive_cramming_trainer import ProgressiveCrammingTrainer

__all__ = [
    "CompressionHeadTrainer",
    "FullCrammingTrainer",
    "LowDimTrainer",
    "PrefixTuningTrainer",
    "ProgressiveCrammingTrainer",
]
