"""Common arguments for evaluation scripts (HellaSwag / ARC / MMLU / generation)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalArgs:
    """Common evaluation arguments shared by HellaSwag, ARC, MMLU and generation.

    Each downstream evaluation script also defines its own benchmark-specific
    arguments (e.g. `--subject` for MMLU); only the truly shared fields live here.
    """

    embeddings_dataset_path: str = field(
        metadata={"help": "Path to the compressed_prefixes / progressive_prefixes dataset on disk."},
    )
    output_file: str | None = field(
        default=None,
        metadata={"help": "Where to save evaluation results (default: alongside the embeddings dataset)."},
    )
    limit_samples: int | None = field(
        default=None,
        metadata={"help": "Optional cap on the number of compressed samples to evaluate."},
    )
    only_full_convergence: bool = field(
        default=False,
        metadata={"help": "Restrict evaluation to samples whose final_convergence is exactly 1.0."},
    )
    text_contains: str | None = field(
        default=None,
        metadata={"help": "Filter compressed samples whose decoded text contains this substring."},
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Evaluation batch size."},
    )
    sample_id: int | None = field(
        default=None,
        metadata={"help": "Restrict evaluation to a single sample_id (debug helper)."},
    )
    stage_index: int | None = field(
        default=None,
        metadata={"help": "Filter to a specific progressive stage (only meaningful for progressive datasets)."},
    )
