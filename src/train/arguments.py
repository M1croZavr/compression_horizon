from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class MyTrainingArguments(TrainingArguments):
    model_checkpoint: str = field(default="HuggingFaceTB/SmolLM2-135M")
    max_optimization_steps_per_sample: int = field(default=10_000)

    ddp_find_unused_parameters: bool = field(default=False)
    load_best_model_at_end: bool = field(default=False)

    number_of_eos_tokens: int = field(default=1)

    learning_rate: float = field(default=1e-4)
    max_grad_norm: float = field(default=1.0)

    per_device_train_batch_size: int = field(default=1)

    weight_decay: float = field(default=0.0)

    dataloader_drop_last: bool = field(default=True)
    dataloader_num_workers: int = field(default=0)
