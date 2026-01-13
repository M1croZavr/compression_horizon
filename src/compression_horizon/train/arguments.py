from __future__ import annotations

from dataclasses import dataclass, field

from transformers import SchedulerType, TrainingArguments


@dataclass
class MyTrainingArguments(TrainingArguments):
    """Training arguments for tokens compression experiments."""

    # Core compression arguments
    model_checkpoint: str = field(
        default="HuggingFaceTB/SmolLM2-135M",
        metadata={"help": "Huggingface location for a model and a tokenizer."},
    )
    low_dim_projection: bool = field(
        default=False,
        metadata={"help": "Low dim projection flag"},
    )
    low_dim_size: int = field(
        default=32,
        metadata={"help": "Dimension of small space for embeddings regularization"},
    )
    low_dim_lr_scheduler_type: SchedulerType | str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."},
    )
    low_dim_warmup_steps: int = field(default=100, metadata={"help": "Linear warmup over warmup_steps."})

    embedding_init_method: str = field(
        default="random",
        metadata={"help": 'Initialization method for compression embeddings: "random", "mvnormal", "pretrained_pca"'},
    )
    pretrained_pca_num_components: int = field(
        default=16,
        metadata={"help": "Number of PCA components to use when embedding_init_method=pretrained_pca."},
    )
    pretrained_pca_path: str = field(
        default="",
        metadata={
            "help": "Path to progressive_prefixes dataset for PCA initialization (when embedding_init_method=pretrained_pca)."
        },
    )
    number_of_mem_tokens: int = field(
        default=1,
        metadata={"help": "Number of trainable [mem] tokens for each sample."},
    )
    loss_type: str = field(
        default="l2",
        metadata={"help": "Loss type for activation alignment: l2, l1, or cosine."},
    )
    hybrid_alpha: float | None = field(
        default=None,
        metadata={
            "help": "Multiplier in the loss function for l2, l1, or cosine, hybrid loss applied in training when specified."
        },
    )
    num_alignment_layers: int = field(default=0, metadata={"help": "Number of transformer layers to align (0 = all)."})
    inverted_alignment: bool = field(
        default=False,
        metadata={
            "help": "Direction of taking transformer layers, True - from depth to shallow, False - from shallow to depth."
        },
    )
    max_sequence_length: int = field(
        default=128,
        metadata={"help": "Max sequence length for compressing in training."},
    )
    max_optimization_steps_per_sample: int = field(
        default=1_000,
        metadata={"help": "Max optimization steps for training 1 sample."},
    )
    random_seed: int | None = field(default=42, metadata={"help": "Random seed for reproducibility (None to skip)."})
    fix_position_ids: bool = field(
        default=False,
    )
    generate_in_compute_loss: bool = field(
        default=False,
    )
    limit_dataset_items: int | None = field(default=1)

    # Overrides with changed defaults
    optim: str = field(
        default="adamw_torch",
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device accelerator core/CPU for training."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    dataloader_drop_last: bool = field(
        default=True,
        metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."},
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )
    learning_rate: float = field(default=1e-2, metadata={"help": "The initial learning rate for an optimizer."})
    adam_beta1: float = 0.9
    adam_beta2: float = 0.9
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay for an optimizer if we apply some."},
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    lr_scheduler_type: SchedulerType | str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."},
    )
    ddp_find_unused_parameters: bool | None = field(
        default=False,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to load the best model found during training at the end of training. When this option"
                " is enabled, the best checkpoint will always be saved. See `save_total_limit` for more."
            )
        },
    )

    # Progressive training control
    noop_train: bool = field(default=False, metadata={"help": "Whether to use noop training."})
    noop_convergence_threshold: float = field(
        default=1.0,
        metadata={"help": "Mean token-level match ratio required to mark a stage as converged."},
    )

    progressive_train: bool = field(default=False, metadata={"help": "Whether to use progressive training."})
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
    save_progressive_artifacts: bool = field(
        default=True,
        metadata={"help": "Whether to persist intermediate compression tokens for each stage."},
    )
    max_tokens_in_distribution: int = field(
        default=1,
        metadata={"help": "Number of top tokens to keep in the distribution target (for train_noop)."},
    )
    # Precision control
    dtype: str = field(
        default="bf16",
        metadata={
            "help": (
                "Torch dtype for model and training. "
                "One of: auto, float32|fp32, bfloat16|bf16, float16|fp16. "
                "This overrides the torch_dtype used to load the model."
            )
        },
    )
