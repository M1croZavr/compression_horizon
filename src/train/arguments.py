from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class MyTrainingArguments(TrainingArguments):
    model_checkpoint: str = field(default="HuggingFaceTB/SmolLM2-135M")
    max_optimization_steps_per_sample: int = field(default=10_000)

    ddp_find_unused_parameters: bool = field(default=False)
    load_best_model_at_end: bool = field(default=False)

    max_sequence_length: int = field(default=128, metadata={"help": "Max sequence length for training"})

    # Loss across hidden states: one of {"l2", "l1", "cosine"}
    loss_type: str = field(default="l2", metadata={"help": "Loss type for activation alignment: l2, l1, or cosine"})
    # If > 0, align only the last N hidden-sta,te layers; 0 means all layers
    num_alignment_layers: int = field(default=0, metadata={"help": "Number of last layers to align (0 = all)"})

    learning_rate: float = field(default=1e-3)
    max_grad_norm: float = field(default=1.0)
    lr_scheduler_type: str = field(default="cosine")

    per_device_train_batch_size: int = field(default=1)

    weight_decay: float = field(default=0.0)

    dataloader_drop_last: bool = field(default=True)
    dataloader_num_workers: int = field(default=0)
