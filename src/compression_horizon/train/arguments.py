from __future__ import annotations

from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class MyTrainingArguments(TrainingArguments):
    """Training arguments for tokens compression experiments."""
    model_checkpoint: str = field(default="HuggingFaceTB/SmolLM2-135M")
    loss_type: str = field(default="l2", metadata={"help": "Loss type for activation alignment: l2, l1, or cosine"})
    max_sequence_length: int = field(default=128, metadata={"help": "Max sequence length for training"})
    random_seed: int | None = field(default=42, metadata={"help": "Random seed for reproducibility (None to skip)"})
    embedding_init_method: str = field(
        default="random",
        metadata={"help": "Initialization method for compression embeddings: random or mvnormal"},
    )
    number_of_mem_tokens: int = field(default=1, metadata={"help": "Number of trainable [mem] tokens for a sample"})
    per_device_train_batch_size: int = field(default=1)
    learning_rate: float = field(default=1e-3)
    lr_scheduler_type: str = field(default="cosine")
    max_optimization_steps_per_sample: int = field(default=10_000)
    num_alignment_layers: int = field(
        default=0,
        metadata={"help": "Number of last layers hidden states to align (0 = all)"}
    )
    inverted_alignment: bool = field(default=False)
    hybrid_alpha: float | None = field(
        default=None,
        metadata={"help": "Multiplier in the loss function for l2, l1, or cosine, hybrid loss applied in training when specified"},
    )

    ddp_find_unused_parameters: bool = field(default=False)
    load_best_model_at_end: bool = field(default=False)
    max_grad_norm: float = field(default=1.0)
    weight_decay: float = field(default=0.0)
    dataloader_drop_last: bool = field(default=True)
    dataloader_num_workers: int = field(default=0)
    # Progressive training control
    progressive_train: bool = field(default=False, metadata={"help": "Whether to use progressive training"})
    progressive_min_seq_len: int = field(
        default=1,
        metadata={"help": "Starting effective sequence length for progressive_train"},
    )
    progressive_step: int = field(
        default=1,
        metadata={"help": "Step size to increase effective sequence length between stages"},
    )
    progressive_convergence_threshold: float = field(
        default=0.99,
        metadata={"help": "Mean token-level match ratio required to mark a stage as converged"},
    )
    progressive_max_stages: int = field(
        default=0,
        metadata={"help": "Optional cap on number of progressive stages (0 = no cap)"},
    )
    save_progressive_artifacts: bool = field(
        default=True,
        metadata={"help": "Whether to persist intermediate compression tokens for each stage"},
    )
