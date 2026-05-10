"""Progressive cramming control arguments (Section 4.1 of the paper)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProgressiveArgs:
    """Progressive cramming controls (paper Section 4.1).

    Enables the per-sample stage-wise procedure that grows the target prefix
    token-by-token and warm-starts each stage from the previous solution.
    """

    progressive_train: bool = field(
        default=False,
        metadata={"help": "Whether to use progressive training."},
    )
    progressive_min_seq_len: int = field(
        default=1,
        metadata={"help": "Starting effective sequence length for progressive_train."},
    )
    progressive_step: int = field(
        default=1,
        metadata={"help": "Step size to increase effective sequence length between stages."},
    )
    progressive_convergence_threshold: float = field(
        default=1.0,
        metadata={"help": "Mean token-level match ratio required to mark a stage as converged."},
    )
    progressive_max_stages: int = field(
        default=0,
        metadata={"help": "Optional cap on number of progressive stages (0 = no cap)."},
    )
    progressive_reset_lr_scheduler_on_non_convergence: bool = field(
        default=False,
        metadata={
            "help": ("If True, reset LR scheduler and continue training when convergence " "fails (only once per stage).")
        },
    )
    max_optimization_steps_per_token: int = field(
        default=1_000,
        metadata={"help": "Max optimization steps for training 1 token (only applicable for progressive training)."},
    )
    save_progressive_artifacts: bool = field(
        default=True,
        metadata={"help": "Whether to persist intermediate compression tokens for each stage."},
    )
