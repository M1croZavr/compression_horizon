from tqdm.auto import tqdm

import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import get_scheduler


class MyTrainer:

    def __init__(
        self,
        model=None,
        processing_class=None,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
    ):
        self.model = model
        self.processing_class = processing_class
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

    def compute_loss(
        self,
        model,
        input_ids,
        inputs_embeds,
        attention_mask,
        model_tokens_with_compression_tokens,
        attention_mask_with_compression_tokens,
        num_compression_tokens,
    ):

        with torch.no_grad():
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        compression_outputs = model(
            inputs_embeds=model_tokens_with_compression_tokens,
            attention_mask=attention_mask_with_compression_tokens,
            output_hidden_states=True,
        )

        loss = 0
        total_layers = len(outputs.hidden_states)
        if getattr(self.args, "num_alignment_layers", 0) and self.args.num_alignment_layers > 0:
            num_layers = min(self.args.num_alignment_layers, total_layers)
            layer_indices = range(total_layers - num_layers, total_layers)
        else:
            layer_indices = range(total_layers)

        loss_type = getattr(self.args, "loss_type", "l2").lower()

        if loss_type == "cross_entropy":
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            loss = F.cross_entropy(compression_outputs.logits[:, 1:].flatten(0, 1), labels.flatten(), reduction="mean")
        else:
            for i in layer_indices:
                tgt = outputs.hidden_states[i]
                pred = compression_outputs.hidden_states[i][:, num_compression_tokens:]
                if loss_type == "l2":
                    loss = loss + F.mse_loss(tgt, pred, reduction="mean")
                elif loss_type == "l1":
                    loss = loss + F.l1_loss(tgt, pred, reduction="mean")
                elif loss_type == "cosine":
                    cos = F.cosine_similarity(tgt, pred, dim=-1)
                    loss = loss + (1.0 - cos).mean()
                else:
                    raise ValueError(f"Unsupported loss_type: {self.args.loss_type}")

        convergece_per_sample = (compression_outputs.logits[:, 1:-1].argmax(dim=-1) == input_ids[:, 1:]).sum(
            dim=-1
        ) / attention_mask.sum(dim=-1)

        return loss, convergece_per_sample.detach().clone()

    def train(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)

        model = self.model.to(device)
        # Freeze model parameters; we only optimize the compression tokens
        for p in model.parameters():
            p.requires_grad_(False)

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

        num_compression_tokens = getattr(self.args, "number_of_eos_tokens", 1)

        for batch in dataloader:
            batch_size = batch["input_ids"].shape[0]
            input_ids = batch.input_ids.squeeze(1)
            # Do not track graph for token embeddings; the model is frozen
            with torch.no_grad():
                model_token_embeddings = model.model.embed_tokens(input_ids)
            attention_mask = batch.attention_mask.squeeze(1)

            # Trainable compression tokens per sample
            compression_tokens = torch.nn.Parameter(
                torch.rand([batch_size, num_compression_tokens, model_token_embeddings.shape[-1]])
            )
            compression_tokens_attention_mask = torch.tensor([[1]], dtype=attention_mask.dtype).repeat(
                batch_size, num_compression_tokens
            )

            optimizer = AdamW([compression_tokens], lr=self.args.learning_rate, weight_decay=0.01)
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.max_optimization_steps_per_sample,
            )

            pbar = tqdm(range(self.args.max_optimization_steps_per_sample), total=self.args.max_optimization_steps_per_sample)
            pbar.set_description("Training")
            for i in pbar:
                # Rebuild concatenations each step to avoid reusing the same autograd graph
                model_tokens_with_compression_tokens = torch.cat([model_token_embeddings, compression_tokens], dim=1)
                attention_mask_with_compression_tokens = torch.cat([attention_mask, compression_tokens_attention_mask], dim=1)
                loss, convergece_per_sample = self.compute_loss(
                    model,
                    input_ids,
                    model_token_embeddings,
                    attention_mask,
                    model_tokens_with_compression_tokens,
                    attention_mask_with_compression_tokens,
                    num_compression_tokens,
                )
                loss.backward()
                pbar.update(1)
                pbar.set_postfix(
                    loss=loss.item(),
                    convergece_per_sample=convergece_per_sample.mean().item(),
                    compression_tokens_mean=compression_tokens.mean().item(),
                    compression_tokens_std=compression_tokens.std().item(),
                    grad=compression_tokens.grad.norm(2).item(),
                    lr=lr_scheduler.get_last_lr()[0],
                )

                # if i == 100:
                #     breakpoint()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
