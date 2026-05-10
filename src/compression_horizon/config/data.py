"""Dataset-loading and tokenization arguments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataArgs:
    """How to load and tokenize the corpus to compress / evaluate on.

    Mirrors the dataset-related fields previously defined directly on
    `MyTrainingArguments`. Used by training and evaluation scripts.
    """

    dataset_name: str = field(
        default="mrsndmn/pg19",
        metadata={"help": "Dataset name to use for training (e.g., 'mrsndmn/pg19')."},
    )
    max_sequence_length: int = field(
        default=128,
        metadata={"help": "Max sequence length for compressing in training."},
    )
    limit_dataset_items: int | None = field(
        default=1,
        metadata={"help": "Optional cap on number of dataset rows to use."},
    )
    offset_dataset_items: int | None = field(
        default=None,
        metadata={"help": "Offset for dataset items selection (applied before limit_dataset_items)."},
    )
    no_bos_token: bool = field(
        default=False,
        metadata={"help": "Disable BOS token insertion during dataset tokenization."},
    )
