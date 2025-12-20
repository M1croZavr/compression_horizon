import hashlib
import os
import subprocess
import sys

import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from compression_horizon.train.arguments import MyTrainingArguments
from compression_horizon.train.trainer import MyTrainer
from compression_horizon.utils.exceptions import NvidiaSMIError
from compression_horizon.utils.launch import resolve_torch_dtype

if __name__ == "__main__":
    # Check for nvidia-smi availability
    try:
        subprocess.check_output(["nvidia-smi"], shell=True)
    except subprocess.CalledProcessError:
        raise NvidiaSMIError("nvidia-smi is not available")

    # Parse command-line arguments and defaults
    hf_parser = transformers.HfArgumentParser(MyTrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()

    # Determine output directory:
    # - If user provided --output_dir, respect it.
    # - Otherwise, construct: artifacts/{experiments|experiments_progressive}/
    #   ch_{essential_params}_{hash8}, where hash8 is derived from training args.
    default_base = "artifacts/experiments_progressive" if training_args.progressive_train else "artifacts/experiments"
    os.makedirs(default_base, exist_ok=True)
    # Build short, human-readable prefix
    loss_type = training_args.loss_type
    hybrid_alpha = training_args.hybrid_alpha
    prefix = (
        f"model_{training_args.model_checkpoint.replace('/', '_')}_mem_{training_args.number_of_mem_tokens}_init_{training_args.embedding_init_method}_seq_len_{training_args.max_sequence_length}"
        if not training_args.hybrid_alpha
        else f"model_{training_args.model_checkpoint.replace('/', '_')}_mem_{training_args.number_of_mem_tokens}_ch_{loss_type}_hybrid_alpha_{hybrid_alpha}_init_{training_args.embedding_init_method}_seq_len_{training_args.max_sequence_length}"
    )
    # If output_dir not provided, compose it using the prefix + args_hash
    output_dir = os.path.join(default_base, f"{prefix}")
    os.makedirs(output_dir, exist_ok=True)
    # Attach to args so trainer can save artifacts there (respecting any user-provided output_dir)
    training_args.output_dir = output_dir
    training_args.logging_dir = output_dir
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

    # Initializing the model and its tokenizer
    torch_dtype = resolve_torch_dtype(training_args.dtype)
    model = AutoModelForCausalLM.from_pretrained(training_args.model_checkpoint, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_checkpoint)
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # Load samples to compress
    raw_dataset = load_dataset("mrsndmn/pg19", split="test", num_proc=4)
    train_dataset = raw_dataset.select(range(training_args.limit_dataset_items))
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
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Train
    transformers.logging.set_verbosity_info()
    trainer = MyTrainer(
        model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    if training_args.progressive_train:
        training_artifacts = trainer.progressive_train()
    else:
        training_artifacts = trainer.train()
    print(f"Saved compressed prefixes to: {training_artifacts}.")
