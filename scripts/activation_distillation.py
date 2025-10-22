import torch
import transformers
from datasets import load_dataset

from train.arguments import MyTrainingArguments
from train.trainer import MyTrainer

from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":

    import subprocess

    subprocess.check_output(["nvidia-smi"])

    hf_parser = transformers.HfArgumentParser(MyTrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()

    model = AutoModelForCausalLM.from_pretrained(training_args.model_checkpoint, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_checkpoint)

    raw_dataset = load_dataset("mrsndmn/pg19", split="test", num_proc=4)
    train_dataset = raw_dataset.select(range(10))
    # eval_dataset = raw_dataset.select(range(10, 20))

    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = train_dataset.map(
        lambda x: tokenizer(
            x["text"], truncation=True, padding="max_length", max_length=training_args.max_sequence_length, return_tensors="pt"
        ),
        remove_columns=train_dataset.column_names,
    )

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

    training_artifacts = trainer.train()
