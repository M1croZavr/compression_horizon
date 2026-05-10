"""Activation alignment loss arguments (Section 4.2 of the paper)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AlignmentArgs:
    """Activation-alignment regularizer (paper Section 4.2, formula 4).

    Controls the auxiliary cosine/L2/L1 loss between hidden states of the
    compressed and uncompressed forward passes, used to stabilize optimization.
    """

    loss_type: str = field(
        default="l2",
        metadata={"help": "Loss type for activation alignment: l2, l1, or cosine."},
    )
    hybrid_alpha: float | None = field(
        default=None,
        metadata={
            "help": (
                "Multiplier in the loss function for l2/l1/cosine alignment. "
                "Hybrid loss is applied in training when specified; "
                "leaving this as None disables alignment."
            )
        },
    )
    num_alignment_layers: int = field(
        default=0,
        metadata={"help": "Number of transformer layers to align (0 = all)."},
    )
    inverted_alignment: bool = field(
        default=False,
        metadata={
            "help": ("Direction of taking transformer layers: " "True = from depth to shallow, False = from shallow to depth.")
        },
    )
