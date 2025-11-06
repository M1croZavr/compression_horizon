import uuid

import torch
import os
import subprocess
import transformers
from datasets import load_dataset

from compression_horizon.train.arguments import MyTrainingArguments
from compression_horizon.train.trainer import MyTrainer

from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer

class NvidiaSMIError(Exception):
    """A custom exception for validating nvidia-smi availability."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


if __name__ == "__main__":
    # Check for nvidia-smi availability
    try:
        subprocess.check_output(["nvidia-smi"], shell=True)
    except subprocess.CalledProcessError:
        raise NvidiaSMIError("nvidia-smi is not available")

    # Parse arguments
    hf_parser = transformers.HfArgumentParser(MyTrainingArguments)
    training_args, = hf_parser.parse_args_into_dataclasses()

    # Make output/logging directory
    output_dir = f"artifacts/experiments/hl_{getattr(training_args, 'loss_type', 'l2')}_{training_args.embedding_init_method}_{uuid.uuid4()}"
    os.makedirs(output_dir, exist_ok=True)
    training_args.output_dir = output_dir
    training_args.logging_dir = output_dir

    # Initialize model and its tokenizer
    model = AutoModelForCausalLM.from_pretrained(training_args.model_checkpoint, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_checkpoint)
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # Load sample to compress
    raw_dataset = load_dataset("mrsndmn/pg19", split="test")
    train_dataset = raw_dataset.select(range(1))
    train_dataset = train_dataset.map(
        lambda x: tokenizer(
            x["text"], truncation=True, padding="max_length", max_length=training_args.max_sequence_length, return_tensors="pt"
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
    print(f"Saved compressed prefixes to: {training_artifacts}")
