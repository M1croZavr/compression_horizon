import os
from typing import Any

import torch
from torch.optim import SGD, AdamW, Optimizer
from transformers import get_scheduler


def build_optimizer_and_scheduler(
    args,
    parameters: list[torch.nn.Parameter],
    num_training_steps: int | None = None,
    num_processes: int = 1,
) -> tuple[Optimizer, Any]:
    """Build the AdamW/SGD optimizer + LR scheduler from training args."""
    if args.optim == "adamw_torch":
        optimizer = AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
        )
    elif args.optim == "sgd":
        optimizer = SGD(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError("Only AdamW and SGD are supported!")

    lr_scheduler = None
    if num_training_steps is not None:
        if args.lr_scheduler_kwargs is not None:
            assert args.lr_scheduler_kwargs["min_lr"] < args.learning_rate, (
                f"min_lr must be lower than regular LR, " f"{args.lr_scheduler_kwargs['min_lr']} < {args.learning_rate}!"
            )
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps * num_processes,
            num_training_steps=num_training_steps * num_processes,
            scheduler_specific_kwargs=args.lr_scheduler_kwargs,
        )
    return optimizer, lr_scheduler


def build_low_dim_projection(args, embedding_dim: int) -> tuple[torch.nn.Linear, Optimizer | None, Any]:
    """Build the [low_dim_size -> hidden] projection module + (optional) optimizer/scheduler."""
    projection = torch.nn.Linear(args.low_dim_size, embedding_dim)

    if args.low_dim_projection_checkpoint is not None:
        if not os.path.exists(args.low_dim_projection_checkpoint):
            raise ValueError(f"low_dim_projection_checkpoint does not exist: {args.low_dim_projection_checkpoint}!")
        checkpoint = torch.load(args.low_dim_projection_checkpoint, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "low_dim_projection" in checkpoint:
                projection.load_state_dict(checkpoint["low_dim_projection"])
            elif "state_dict" in checkpoint:
                projection.load_state_dict(checkpoint["state_dict"])
            else:
                projection.load_state_dict(checkpoint)
        else:
            projection.load_state_dict(checkpoint)

    if not args.low_dim_projection_train:
        for param in projection.parameters():
            param.requires_grad = False
        return projection, None, None

    optimizer = AdamW(
        projection.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_optimization_steps_per_sample,
        scheduler_specific_kwargs=args.lr_scheduler_kwargs,
    )
    return projection, optimizer, scheduler
