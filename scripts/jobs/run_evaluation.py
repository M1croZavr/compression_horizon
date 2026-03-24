#!/usr/bin/env python3
"""
Evaluation job launcher for compression_horizon experiments.

This project does not require separate evaluation jobs — evaluation
metrics are computed inline during training. This script is a no-op
that satisfies the research loop interface.
"""
import argparse
import json


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation jobs (no-op for this project).")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", choices=("text", "json"), default="text")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    if args.output == "json":
        print(json.dumps({"jobs": [], "launched": 0}))
    else:
        print("No evaluation jobs needed for this project.")
