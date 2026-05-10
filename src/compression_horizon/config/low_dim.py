"""Low-dimensional projection arguments (Section 4.3 of the paper)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LowDimArgs:
    """Low-dimensional reparameterization e = W z + b (paper Section 4.3).

    Restricts the compression embedding to an affine rank-k subspace by
    optimizing low-dimensional coefficients z together with an affine map
    W, b instead of the embedding directly.
    """

    low_dim_projection: bool = field(
        default=False,
        metadata={"help": "Enable low-dim projection reparameterization e = W z + b."},
    )
    low_dim_projection_global: bool = field(
        default=False,
        metadata={"help": "If True, share the same projection across the dataset (vs. per-batch)."},
    )
    low_dim_size: int = field(
        default=32,
        metadata={"help": "Dimension k of the low-dim subspace for embedding regularization."},
    )
    low_dim_projection_checkpoint: str | None = field(
        default=None,
        metadata={"help": "Path to checkpoint file to load low-dimensional projection state from."},
    )
    low_dim_projection_train: bool = field(
        default=True,
        metadata={"help": "Whether to optimize the low-dimensional projection (False to freeze it)."},
    )
