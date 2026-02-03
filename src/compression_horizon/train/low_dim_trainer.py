"""Low-dimensional compression trainer: compression in low-dim space with projection."""

import os

import torch
from tqdm.auto import tqdm

from compression_horizon.train.base import BaseTrainer
from compression_horizon.utils.launch import freeze_model_parameters, get_device, set_launch_seed


class LowDimTrainer(BaseTrainer):
    """Trainer for low-dimensional compression tokens with a projection to hidden size."""

    def train(self) -> str | None:
        """Run low-dim training. Returns save path or None."""
        return self._train_low_dim()

    def _train_low_dim(self) -> str | None:
        print("Train low dim!!")

        set_launch_seed(self.args.random_seed)
        device = get_device()
        model = self.model.to(device)
        freeze_model_parameters(model)
        init_method, mvn_dist, pca_components, pca_mean, loaded_embeddings = self._prepare_embedding_init(model)
        num_compression_tokens = self.args.number_of_mem_tokens

        collected_rows = []
        sample_id_counter = 0

        hidden_size = model.config.hidden_size

        dataloader = self._create_dataloader()
        final_projection = None
        for batch in tqdm(dataloader):
            model.eval()
            input_ids = batch.input_ids.squeeze(1).to(device)
            batch_size = input_ids.shape[0]

            attention_mask = batch.attention_mask.squeeze(1).to(device)
            with torch.no_grad():
                token_embeddings = model.get_input_embeddings()(input_ids)

            target_hidden = self.compute_target_hidden(model, token_embeddings, attention_mask)

            compression_token_embeddings = self._init_compression_tokens(
                batch_size,
                num_compression_tokens,
                self.args.low_dim_size,
                init_method,
                mvn_dist,
                single_compressed_embeddings_initialization=None,
                token_embeddings=token_embeddings,
                pca_components=pca_components,
                pca_mean=pca_mean,
                loaded_embeddings=loaded_embeddings,
            )

            (
                projection,
                projection_optimizer,
                projection_lr_scheduler,
            ) = self._prepare_low_dim_proj(embedding_dim=hidden_size)
            projection = projection.to(device)
            print("projection_optimizer", projection_optimizer)

            compression_token_embeddings = torch.nn.Parameter(compression_token_embeddings.data.to(device))
            initialization_embeddings = compression_token_embeddings.detach().clone().cpu()
            optimizer, lr_scheduler = self._build_optimizer_and_scheduler(
                [compression_token_embeddings],
                num_training_steps=self.args.max_optimization_steps_per_sample,
            )

            compression_attention_mask = torch.tensor([1], dtype=attention_mask.dtype, device=device).repeat(
                batch_size, num_compression_tokens
            )

            (
                loss,
                alignment_loss,
                convergence_per_sample,
                generated_text,
                ground_truth_text,
            ) = (None, None, None, None, None)
            progress_bar = tqdm(
                range(self.args.max_optimization_steps_per_sample),
                total=self.args.max_optimization_steps_per_sample,
            )
            progress_bar.set_description("Training")

            total_per_sample_convergence = torch.zeros(
                [
                    self.args.max_optimization_steps_per_sample,
                    input_ids.shape[0],
                ],
                dtype=torch.long,
            )
            total_per_sample_convergence_099 = torch.zeros(
                [
                    self.args.max_optimization_steps_per_sample,
                    input_ids.shape[0],
                ],
                dtype=torch.long,
            )
            total_per_sample_convergence_095 = torch.zeros(
                [
                    self.args.max_optimization_steps_per_sample,
                    input_ids.shape[0],
                ],
                dtype=torch.long,
            )

            for step_i in progress_bar:
                compression_token_embeddings_llm = projection(compression_token_embeddings)

                united_token_embeddings = torch.cat(
                    [
                        compression_token_embeddings_llm.to(token_embeddings.device).to(token_embeddings.dtype),
                        token_embeddings,
                    ],
                    dim=1,
                )
                united_attention_mask = torch.cat(
                    [compression_attention_mask, attention_mask],
                    dim=1,
                )
                (
                    loss,
                    alignment_loss,
                    convergence_per_sample,
                    generated_text,
                    ground_truth_text,
                ) = self.compute_loss(
                    model,
                    input_ids,
                    token_embeddings,
                    attention_mask,
                    united_token_embeddings,
                    united_attention_mask,
                    num_compression_tokens,
                    target_hidden=target_hidden,
                )
                loss.backward()

                optimizer.step()
                if projection_optimizer is not None:
                    projection_optimizer.step()

                with torch.no_grad():
                    progress_bar.update(1)
                    alignment_loss_item = alignment_loss.item() if alignment_loss is not None else None
                    progress_bar.set_postfix(
                        loss=loss.item(),
                        loss_alignment=alignment_loss_item,
                        convergece_per_sample=convergence_per_sample.mean().item(),
                        lr=lr_scheduler.get_last_lr()[0],
                    )
                    self._log_step(
                        loss,
                        alignment_loss,
                        convergence_per_sample,
                        compression_token_embeddings,
                        lr_scheduler,
                        generated_text,
                        ground_truth_text,
                    )

                    if (convergence_per_sample == 1.0).all():
                        print(f"Early stopping: compression converged in {step_i} steps")
                        break

                total_per_sample_convergence[step_i, :] = convergence_per_sample < 1.0
                total_per_sample_convergence_099[step_i, :] = convergence_per_sample < 0.99
                total_per_sample_convergence_095[step_i, :] = convergence_per_sample < 0.95

                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                if projection_optimizer is not None:
                    projection_optimizer.zero_grad(set_to_none=True)
                if projection_lr_scheduler is not None:
                    projection_lr_scheduler.step()

            total_per_sample_convergence_sum = total_per_sample_convergence.sum(dim=0)
            total_per_sample_convergence_099_sum = total_per_sample_convergence_099.sum(dim=0)
            total_per_sample_convergence_095_sum = total_per_sample_convergence_095.sum(dim=0)

            with torch.no_grad():
                tokenizer = self.processing_class
                last_loss = loss.item()
                last_convergence_per_sample = convergence_per_sample.cpu()
                pca_coefficients_to_save = None
                compression_token_embeddings_cpu = compression_token_embeddings.detach().cpu()
                for j in range(batch_size):
                    sample_attention_mask = attention_mask[j].bool()
                    sample_input_ids = input_ids[j][sample_attention_mask]
                    sample_text = tokenizer.decode(sample_input_ids, skip_special_tokens=True)
                    embedding = compression_token_embeddings_cpu[j].to(torch.float32).numpy().tolist()
                    initialization_embedding = initialization_embeddings[j].to(torch.float32).numpy().tolist()
                    compression_token_embeddings_mean = float(compression_token_embeddings_cpu[j].mean().item())
                    compression_token_embeddings_std = float(compression_token_embeddings_cpu[j].std().item())
                    item_convergence_per_sample = total_per_sample_convergence_sum[j].item()
                    collected_rows.append(
                        {
                            "sample_id": sample_id_counter,
                            "text": sample_text,
                            "embedding": embedding,
                            "pca_coefficients": (pca_coefficients_to_save[j] if pca_coefficients_to_save is not None else None),
                            "initialization_embedding": initialization_embedding,
                            "final_loss": last_loss,
                            "final_convergence": last_convergence_per_sample[j].item(),
                            "convergence_after_steps": item_convergence_per_sample,
                            "convergence_0.99_after_steps": int(total_per_sample_convergence_099_sum[j].item()),
                            "convergence_0.95_after_steps": int(total_per_sample_convergence_095_sum[j].item()),
                            "compression_tokens_mean": compression_token_embeddings_mean,
                            "compression_tokens_std": compression_token_embeddings_std,
                            "num_input_tokens": int(sample_attention_mask.sum().item()),
                            "num_compression_tokens": int(num_compression_tokens),
                            "hidden_size": hidden_size,
                            "fix_position_ids": self.args.fix_position_ids,
                            "loss_type": self.args.loss_type,
                            "hybrid_alpha": self.args.hybrid_alpha,
                            "dtype": self.args.dtype,
                            "embedding_init_method": self.args.embedding_init_method,
                            "num_alignment_layers": self.args.num_alignment_layers,
                            "model_checkpoint": self.args.model_checkpoint,
                            "max_optimization_steps_per_sample": self.args.max_optimization_steps_per_sample,
                        }
                    )
                    sample_id_counter += 1
                    final_compression_token_embeddings_cpu = compression_token_embeddings_cpu

            final_projection = projection

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        if final_projection is not None and self.args.low_dim_proj_train:
            output_dir = self.args.output_dir
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                projection_save_path = os.path.join(output_dir, "low_dim_projection.pt")
                torch.save(
                    {
                        "low_dim_projection": final_projection.state_dict(),
                        "low_dim_size": self.args.low_dim_size,
                        "hidden_size": hidden_size,
                    },
                    projection_save_path,
                )
                print(f"Saved low-dimensional projection weights to {projection_save_path}")

        save_path = self._save_artifacts(
            final_compression_token_embeddings_cpu,
            collected_rows,
            "compressed_prefixes",
        )
        if save_path is not None:
            return save_path
        return None
