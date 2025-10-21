from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

@dataclass
class MyTrainingArguments(TrainingArguments):

    ddp_find_unused_parameters: bool = field(default=False)
    load_best_model_at_end: bool = field(default=False)

    number_of_eos_tokens: int = field(default=1)

    learning_rate: float = field(default=1e-4)
    max_grad_norm: float = field(default=1.0)

    warmup_steps: int = field(default=100)
    per_device_train_batch_size: int = field(default=32)

    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    lr_scheduler_kwargs: Optional[dict] = field(default_factory=lambda: {'min_lr': 5e-5})

    average_tokens_across_devices: bool = field(default=True)

    model_checkpoint: str = field(default="HuggingFaceTB/SmolLM2-135M")

    weight_decay: float = field(default=0.0)
    eval_strategy: str = field(default="no")

    save_strategy: str = field(default="no")

    save_total_limit: Optional[int] = field(default=1)
    save_only_model: bool = field(default=True)

    push_to_hub: bool = field(default=False)

    optim: str = field(default="adamw_torch_fused")

    report_to: str = field(default="tensorboard")  # clearml | wandb | none | tensorboard
    logging_steps: int = field(default=1)

    dataloader_drop_last: bool = field(default=True)
    dataloader_num_workers: int = field(default=0)
    bf16: bool = field(default=False)

    torch_compile: bool = field(default=False)

    # dataset: str = field(default="fineweb_edu")  # fineweb_edu | dclm | my_recall

