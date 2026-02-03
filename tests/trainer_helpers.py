"""Shared test helpers for trainer tests."""

from dataclasses import replace

import torch
from torch.utils.data import Dataset

from compression_horizon.train.arguments import MyTrainingArguments


def _make_args(**overrides):
    training_args = MyTrainingArguments()
    defaults = dict(
        model_checkpoint="dummy",
        max_optimization_steps_per_sample=1,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=False,
        max_sequence_length=8,
        loss_type="cross_entropy",
        num_alignment_layers=0,
        learning_rate=1e-1,
        max_grad_norm=1.0,
        lr_scheduler_type="constant",
        lr_scheduler_kwargs=None,
        per_device_train_batch_size=1,
        weight_decay=0.0,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        warmup_steps=0,
        logging_dir=None,
        number_of_mem_tokens=1,
    )
    training_args = replace(training_args, **defaults)
    training_args = replace(training_args, **overrides)
    return training_args


class TinyDataset(Dataset):
    """Minimal dataset for trainer smoke tests."""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long, device="cuda")
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class MultiAccessBatch:
    """Batch that supports both dict-style and attribute access."""

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError as e:
            raise AttributeError(str(e))


def _collate_batch(samples):
    """Stack samples into [B, 1, L] to mirror tokenizer return_tensors."""
    input_ids = torch.stack([s["input_ids"] for s in samples], dim=0).unsqueeze(1)
    attention_mask = torch.stack([s["attention_mask"] for s in samples], dim=0).unsqueeze(1)
    return MultiAccessBatch({"input_ids": input_ids, "attention_mask": attention_mask})
