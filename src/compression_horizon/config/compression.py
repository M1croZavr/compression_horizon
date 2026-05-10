"""Compression-token shape and initialization arguments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CompressionArgs:
    """Defines the compression embedding(s) to be optimized.

    Used by every script that handles compression embeddings: training,
    progressive training, evaluation, generation.
    """

    number_of_mem_tokens: int = field(
        default=1,
        metadata={"help": "Number of trainable [mem] tokens for a single sample."},
    )
    embedding_init_method: str = field(
        default="random",
        metadata={
            "help": (
                'Initialization method for compression embeddings: "random", "mvnormal", '
                '"pretrained_pca", "load_from_disk", and the various "random*" / "neg_random*" / '
                '"random_norm*" / "single_*" variants registered in train/base.py.'
            )
        },
    )
    embedding_init_path: str = field(
        default="",
        metadata={
            "help": (
                "Path to file containing initial compression embeddings "
                "(when embedding_init_method=load_from_disk). "
                "If empty, embeddings will be generated using "
                "load_from_disk_embedding_init_method and saved. "
                "File should contain a tensor of shape [num_tokens, hidden_size] "
                "or [1, num_tokens, hidden_size] or "
                "[batch_size, num_tokens, hidden_size]."
            )
        },
    )
    load_from_disk_embedding_init_method: str = field(
        default="random",
        metadata={
            "help": (
                "Initialization method to use when generating embeddings for "
                "load_from_disk (when embedding_init_path is empty). "
                'Can be any valid embedding_init_method value (e.g., "random", '
                '"random0.02", "zeros", etc.).'
            )
        },
    )
    pretrained_pca_num_components: int = field(
        default=16,
        metadata={"help": "Number of PCA components to use when embedding_init_method=pretrained_pca."},
    )
    pretrained_pca_path: str = field(
        default="",
        metadata={
            "help": (
                "Path to progressive_prefixes dataset for PCA initialization " "(when embedding_init_method=pretrained_pca)."
            )
        },
    )
    fix_position_ids: bool = field(
        default=False,
        metadata={"help": "Whether position_ids should be adjusted relative to compression embeddings."},
    )
