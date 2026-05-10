"""Composable CLI argument groups for compression_horizon scripts.

Each module in this package defines a single dataclass that captures one
orthogonal aspect of an experiment. Scripts compose the groups they need via
``transformers.HfArgumentParser``::

    from compression_horizon.config import (
        ModelArgs, DataArgs, CompressionArgs, EvalArgs,
    )
    from transformers import HfArgumentParser

    parser = HfArgumentParser((ModelArgs, DataArgs, CompressionArgs, EvalArgs))
    model_args, data_args, comp_args, eval_args = parser.parse_args_into_dataclasses()

These dataclasses do **not** inherit from ``transformers.TrainingArguments`` —
the heavy training-runtime fields stay on
:class:`compression_horizon.train.arguments.MyTrainingArguments` for now.
The new dataclasses are intended for evaluation / analysis / generation scripts
and as building blocks for a future composition-based ``MyTrainingArguments``.
"""

from compression_horizon.config.alignment import AlignmentArgs
from compression_horizon.config.compression import CompressionArgs
from compression_horizon.config.data import DataArgs
from compression_horizon.config.eval import EvalArgs
from compression_horizon.config.low_dim import LowDimArgs
from compression_horizon.config.model import ModelArgs
from compression_horizon.config.progressive import ProgressiveArgs

__all__ = [
    "AlignmentArgs",
    "CompressionArgs",
    "DataArgs",
    "EvalArgs",
    "LowDimArgs",
    "ModelArgs",
    "ProgressiveArgs",
]
