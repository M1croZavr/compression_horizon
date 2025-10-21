import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GenerationConfig, Trainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model


class MyTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        log_metrics=True,
        log_prefix="debug",
        force_log=False,
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        # if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:

        labels = inputs.pop("labels")
        # print("labels", (labels != -100).sum())
        # breakpoint()

        unwrapped_model = self.accelerator.unwrap_model(model)

        # Optionally disable loss on EOS tokens when using multiple EOS tokens

        attention_mask = inputs["attention_mask"]
        # token_frequency = inputs.get('token_frequency', None)
        model_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": attention_mask,
            "use_cache": False,
            "output_attentions": False,
        }

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            model_kwargs = {**model_kwargs, **loss_kwargs}

        outputs = model(**model_kwargs)
        # [ bs, seq_len, 2 ]

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None and self.label_smoother is not None or self.compute_loss_func is not None:
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()

            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(
                    outputs.logits, labels, vocab_size=unwrapped_model.config.vocab_size, num_items_in_batch=num_items_in_batch
                )
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and (self.model_accepts_loss_kwargs or self.compute_loss_func):
            loss *= self.accelerator.num_processes

        outputs.loss = loss

        return (loss, outputs) if return_outputs else loss

