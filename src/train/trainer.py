import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW


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
        for i in range(len(outputs.hidden_states)):
            loss += F.mse_loss(
                outputs.hidden_states[i],
                compression_outputs.hidden_states[i][:, num_compression_tokens:],
                reduction="mean",
            )

        convergece_per_sample = (compression_outputs.logits[:, 1:-1].argmax(dim=-1) == input_ids[:, 1:]).sum(
            dim=-1
        ) / attention_mask.sum(dim=-1)

        return loss, convergece_per_sample.detach().clone()

    def train(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)

        model = self.model.to(device)

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

        num_compression_tokens = 1

        for batch in dataloader:
            batch_size = batch["input_ids"].shape[0]
            input_ids = batch.input_ids.squeeze(1)
            model_token_embeddings = model.model.embed_tokens(input_ids)
            attention_mask = batch.attention_mask.squeeze(1)

            compression_tokens = torch.rand([batch_size, num_compression_tokens, model_token_embeddings.shape[-1]])
            compression_tokens_attention_mask = torch.tensor([[1]]).repeat(batch_size, num_compression_tokens)

            model_tokens_with_compression_tokens = torch.cat([model_token_embeddings, compression_tokens], dim=1)
            attention_mask_with_compression_tokens = torch.cat([attention_mask, compression_tokens_attention_mask], dim=1)

            optimizer = AdamW([compression_tokens], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

            for _ in range(self.args.max_optimization_steps_per_sample):
                loss, convergece_per_sample = self.compute_loss(
                    model,
                    input_ids,
                    model_token_embeddings,
                    attention_mask,
                    model_tokens_with_compression_tokens,
                    attention_mask_with_compression_tokens,
                    num_compression_tokens,
                )
                print("loss", loss.item(), "convergece_per_sample", convergece_per_sample.mean().item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
