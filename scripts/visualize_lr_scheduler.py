#!/usr/bin/env python3
"""Temporary script to visualize learning rate scheduler with CLI arguments."""

import argparse
import json
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import get_scheduler


def parse_lr_scheduler_kwargs(kwargs_str: str | None) -> dict:
    """Parse lr_scheduler_kwargs from JSON string."""
    if kwargs_str is None or kwargs_str == "":
        return {}
    try:
        return json.loads(kwargs_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing lr_scheduler_kwargs: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Visualize learning rate scheduler")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        required=True,
        help="Type of learning rate scheduler (e.g., 'cosine', 'linear', 'constant', etc.)",
    )
    parser.add_argument(
        "--lr_scheduler_kwargs",
        type=str,
        default=None,
        help="Additional keyword arguments for the scheduler as JSON string (e.g., '{\"num_cycles\": 0.5}')",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--num_training_steps",
        type=int,
        required=True,
        help="Total number of training steps",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps (default: 0)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path for the plot (default: show interactively)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for the output plot (default: 100)",
    )

    args = parser.parse_args()

    # Parse lr_scheduler_kwargs
    lr_scheduler_kwargs = parse_lr_scheduler_kwargs(args.lr_scheduler_kwargs)

    # Create a dummy optimizer with a single parameter
    dummy_param = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([dummy_param], lr=args.learning_rate)

    # Build scheduler kwargs
    scheduler_kwargs = {
        "optimizer": optimizer,
        "num_warmup_steps": args.num_warmup_steps,
        "num_training_steps": args.num_training_steps,
    }
    scheduler_kwargs.update(lr_scheduler_kwargs)

    # Get scheduler from transformers
    try:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            **scheduler_kwargs,
        )
    except Exception as e:
        print(f"Error creating scheduler: {e}", file=sys.stderr)
        sys.exit(1)

    # Collect learning rates over all steps
    learning_rates = []
    for step in range(args.num_training_steps):
        # Get current LR
        current_lr = lr_scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        # Step the scheduler (but don't actually optimize)
        lr_scheduler.step()

    # Plot the learning rate schedule
    plt.figure(figsize=(12, 6))
    plt.plot(range(args.num_training_steps), learning_rates, linewidth=2)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.title(
        f"Learning Rate Schedule: {args.lr_scheduler_type}\n"
        f"LR={args.learning_rate}, Warmup={args.num_warmup_steps}, Steps={args.num_training_steps}",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save or show
    if args.output_file:
        plt.savefig(args.output_file, dpi=args.dpi, bbox_inches="tight")
        print(f"Plot saved to {args.output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
