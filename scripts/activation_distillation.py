import os
import shutil
from pathlib import Path

import torch
import transformers
from datasets import Dataset, load_dataset

from train.arguments import MyTrainingArguments
from train.trainer import MyTrainer

from transformers import DataCollatorForLanguageModeling, TrainerCallback, AutoModelForCausalLM, AutoTokenizer
from transformers.loss.loss_utils import ForCausalLMLoss

if __name__ == "__main__":

    import subprocess

    subprocess.check_output(["nvidia-smi"])

    hf_parser = transformers.HfArgumentParser(MyTrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()

    model = AutoModelForCausalLM.from_pretrained(training_args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_checkpoint)

    fw_dataset = load_dataset('HuggingFaceFW/fineweb-edu', 'sample-10BT', split='train', num_proc=4)
    train_dataset = fw_dataset.select(range(10))
    eval_dataset = fw_dataset.select(range(10, 20))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trackers_project_name = os.path.basename(training_args.output_dir)
    training_args.run_name = trackers_project_name

    transformers.logging.set_verbosity_info()

    trainer = MyTrainer(
        model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_loss_func=ForCausalLMLoss,
    )

    trainer.accelerator.init_trackers(
        project_name=trackers_project_name,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
