import hashlib
import json
import os
import subprocess
import sys

import torch
import transformers
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from compression_horizon.train.arguments import MyTrainingArguments
from compression_horizon.train.trainer import MyTrainer


class NvidiaSMIError(Exception):
    """A custom exception for validating nvidia-smi availability."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


if __name__ == "__main__":
    try:
        subprocess.check_output(["nvidia-smi"], shell=True)
    except subprocess.CalledProcessError:
        raise NvidiaSMIError("nvidia-smi is not available")

    hf_parser = transformers.HfArgumentParser(MyTrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()

    def _resolve_torch_dtype(dtype_str: str):
        s = (dtype_str or "").lower()
        if s in {"auto"}:
            return "auto"
        if s in {"float32", "fp32"}:
            return torch.float32
        if s in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if s in {"float16", "fp16"}:
            return torch.float16
        # Fallback to float32 for unknown values
        return torch.float32

    # Determine output directory:
    # - If user provided --output_dir, respect it.
    # - Otherwise, construct: artifacts/{experiments|experiments_progressive}/
    #   ch_{essential_params}_{hash8}, where hash8 is derived from training args.
    default_base = "artifacts/experiments_progressive" if training_args.progressive_train else "artifacts/experiments"
    os.makedirs(default_base, exist_ok=True)

    # Build short, human-readable prefix
    loss_type = getattr(training_args, "loss_type", "l2")
    hybrid_alpha = getattr(training_args, "hybrid_alpha", None)
    prefix = (
        f"ch_{loss_type}_init_{training_args.embedding_init_method}_seq_len_{training_args.max_sequence_length}"
        if training_args.progressive_train
        else f"ch_{loss_type}_hybrid_alpha_{hybrid_alpha}_init_{training_args.embedding_init_method}_seq_len_{training_args.max_sequence_length}"
    )

    # Compute stable hash from training arguments (excluding volatile dirs)
    args_dict = training_args.to_dict()
    args_dict.pop("output_dir", None)
    args_dict.pop("logging_dir", None)
    args_json = json.dumps(args_dict, sort_keys=True, ensure_ascii=False, default=str)

    # If output_dir not provided, compose it using the prefix + args_hash
    output_dir = training_args.output_dir
    if not output_dir:
        output_dir = os.path.join(default_base, f"{prefix}")

    os.makedirs(output_dir, exist_ok=True)

    # Ensure logging_dir is set; default to output_dir if not provided
    if not getattr(training_args, "logging_dir", None):
        training_args.logging_dir = output_dir
    # Attach to args so trainer can save artifacts there (respecting any user-provided output_dir)
    training_args.output_dir = output_dir

    # Also persist raw CLI (excluding --output_dir) and its hash for auditability
    argv = sys.argv[1:]
    filtered_argv: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--output_dir":
            skip_next = True
            continue
        if token.startswith("--output_dir="):
            continue
        filtered_argv.append(token)
    cmdline_str = " ".join(filtered_argv).strip()
    cmd_hash8 = hashlib.sha1(cmdline_str.encode("utf-8")).hexdigest()[:8]
    with open(os.path.join(output_dir, "cmd.txt"), "w", encoding="utf-8") as f:
        f.write(cmdline_str + "\n")
    with open(os.path.join(output_dir, "cmd_hash.txt"), "w", encoding="utf-8") as f:
        f.write(cmd_hash8 + "\n")

    torch_dtype = _resolve_torch_dtype(getattr(training_args, "dtype", "float32"))
    model = AutoModelForCausalLM.from_pretrained(training_args.model_checkpoint, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    # Create cache directory for tokenized datasets
    cache_dir = "artifacts/cache/tokenized_datasets"
    os.makedirs(cache_dir, exist_ok=True)

    # Generate cache key based on dataset parameters
    cache_params = {
        "dataset": "mrsndmn/pg19",
        "split": "test",
        "limit_dataset_items": getattr(training_args, "limit_dataset_items", None),
        "max_sequence_length": training_args.max_sequence_length,
        "model_checkpoint": training_args.model_checkpoint,
    }
    cache_key_json = json.dumps(cache_params, sort_keys=True, ensure_ascii=False, default=str)
    cache_key_hash = hashlib.sha256(cache_key_json.encode("utf-8")).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"dataset_{cache_key_hash}")

    # Try to load cached tokenized dataset
    if os.path.exists(cache_path):
        print(f"Loading tokenized dataset from cache: {cache_path}")
        train_dataset = Dataset.load_from_disk(cache_path)
    else:
        print("Tokenizing dataset (this may take a while)...")
        raw_dataset = load_dataset("mrsndmn/pg19", split="test", num_proc=4)

        if training_args.limit_dataset_items is not None:
            train_dataset = raw_dataset.select(range(training_args.limit_dataset_items))
        else:
            train_dataset = raw_dataset
        # eval_dataset = raw_dataset.select(range(10, 20))

        train_dataset = train_dataset.map(
            lambda x: tokenizer(
                x["text"],
                truncation=True,
                padding="max_length",
                max_length=training_args.max_sequence_length,
                return_tensors="pt",
            ),
            remove_columns=train_dataset.column_names,
        )

        # Save tokenized dataset to cache
        print(f"Saving tokenized dataset to cache: {cache_path}")
        train_dataset.save_to_disk(cache_path)

    print("train_dataset", len(train_dataset))
    print("train_dataset", train_dataset)
    # print("eval_dataset", len(eval_dataset))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    transformers.logging.set_verbosity_info()

    trainer = MyTrainer(
        model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    if training_args.progressive_train:
        training_artifacts = trainer.train()
    elif training_args.noop_train:
        training_artifacts = trainer.train_noop()
    else:
        training_artifacts = trainer.train()
    print(f"Saved compressed prefixes to: {training_artifacts}")
