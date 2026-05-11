"""Analysis utilities for compression embeddings.

Reusable metric / probe / intervention code that's shared between trainers,
evaluation scripts and visualization scripts.
"""

from compression_horizon.analysis.attention_hijacking import (
    compute_attention_mass_profile,
    compute_sample_profiles,
    pearson_correlation,
    summarize_hijacking,
)
from compression_horizon.analysis.convergence import (
    ConvergedSamplesGuard,
    ConvergenceTracker,
    ProgressiveSampleStateMachine,
)
from compression_horizon.analysis.information_gain import compute_information_gain

__all__ = [
    "compute_information_gain",
    "ConvergedSamplesGuard",
    "ConvergenceTracker",
    "ProgressiveSampleStateMachine",
    "compute_attention_mass_profile",
    "compute_sample_profiles",
    "pearson_correlation",
    "summarize_hijacking",
]
