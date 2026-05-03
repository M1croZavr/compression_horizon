"""Full cramming trainer: per-batch compression tokens with alignment loss."""

import math

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from compression_horizon.train.base import BaseTrainer
from compression_horizon.utils.launch import freeze_model_parameters, get_device, set_launch_seed


class FullCrammingTrainer(BaseTrainer):
    """Trainer for full cramming: per-sample compression tokens, alignment + CE loss."""

    def train(self) -> str | None:
        """Run full cramming training. Returns save path or None."""
        return self._train_full_cramming()

    def _train_full_cramming(self) -> str | None:
        set_launch_seed(self.args.random_seed)
        device = get_device()
        model = self.model.to(device)
        freeze_model_parameters(model)
        init_method, mvn_dist, pca_components, pca_mean, loaded_embeddings = self._prepare_embedding_init(model)
        num_compression_tokens = self.args.number_of_mem_tokens

        collected_rows = []
        sample_id_counter = 0

        hidden_size = model.config.hidden_size

        single_compressed_embeddings_initialization = None
        if init_method.startswith("single_"):
            single_compressed_embeddings_initialization = self._init_compression_tokens(
                1,
                num_compression_tokens,
                hidden_size,
                init_method,
                mvn_dist,
                token_embeddings=None,
                single_compressed_embeddings_initialization=None,
                pca_components=pca_components,
                pca_mean=pca_mean,
                loaded_embeddings=loaded_embeddings,
            )
            single_compressed_embeddings_initialization = single_compressed_embeddings_initialization.data.detach().clone()

        dataloader = self._create_dataloader()
        for batch in tqdm(dataloader):
            model.eval()
            input_ids = batch.input_ids.squeeze(1).to(device)
            batch_size = input_ids.shape[0]

            attention_mask = batch.attention_mask.squeeze(1).to(device)
            with torch.no_grad():
                token_embeddings = model.get_input_embeddings()(input_ids)

            if self.args.loss_type != "cross_entropy":
                target_hidden = self.compute_target_hidden(model, token_embeddings, attention_mask)
            else:
                target_hidden = None

            if init_method == "pretrained_pca":
                assert pca_components is not None
                assert pca_mean is not None

                pca_components_device = pca_components.to(device)
                pca_mean_device = pca_mean.to(device)

                flattened_dim = pca_mean_device.shape[0]
                expected_flattened_dim = num_compression_tokens * hidden_size
                if flattened_dim != expected_flattened_dim:
                    raise ValueError(
                        f"PCA dimension mismatch: pretrained has {flattened_dim}, "
                        f"but current needs {expected_flattened_dim} "
                        f"(num_tokens={num_compression_tokens}, hidden_size={hidden_size})"
                    )

                n_components = pca_components_device.shape[0]
                pca_coefficients = torch.nn.Parameter(
                    torch.randn([batch_size, n_components], dtype=torch.float32, device=device) * 0.1
                )

                reconstructed_flat = torch.matmul(pca_coefficients, pca_components_device) + pca_mean_device.unsqueeze(0)
                initialization_embeddings = (
                    reconstructed_flat.reshape(batch_size, num_compression_tokens, hidden_size).detach().cpu()
                )

                optimizer, lr_scheduler = self._build_optimizer_and_scheduler(
                    [pca_coefficients],
                    num_training_steps=self.args.max_optimization_steps_per_sample,
                )
            else:
                compression_token_embeddings = self._init_compression_tokens(
                    batch_size,
                    num_compression_tokens,
                    hidden_size,
                    init_method,
                    mvn_dist,
                    single_compressed_embeddings_initialization=single_compressed_embeddings_initialization,
                    token_embeddings=token_embeddings,
                    pca_components=pca_components,
                    pca_mean=pca_mean,
                    loaded_embeddings=loaded_embeddings,
                )
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
            prev_convergence = None
            total_per_sample_convergence = torch.zeros(
                [self.args.max_optimization_steps_per_sample, batch_size],
                dtype=torch.long,
            )
            total_per_sample_convergence_099 = torch.zeros(
                [self.args.max_optimization_steps_per_sample, batch_size],
                dtype=torch.long,
            )
            total_per_sample_convergence_095 = torch.zeros(
                [self.args.max_optimization_steps_per_sample, batch_size],
                dtype=torch.long,
            )
            progress_bar = tqdm(
                range(self.args.max_optimization_steps_per_sample),
                total=self.args.max_optimization_steps_per_sample,
            )

            print(
                "self.args.max_optimization_steps_per_sample",
                self.args.max_optimization_steps_per_sample,
            )

            progress_bar.set_description("Training")
            for step_i in progress_bar:
                if init_method == "pretrained_pca":
                    reconstructed_flat = torch.matmul(pca_coefficients, pca_components_device) + pca_mean_device.unsqueeze(0)
                    compression_token_embeddings = reconstructed_flat.reshape(batch_size, num_compression_tokens, hidden_size)

                united_token_embeddings = torch.cat(
                    [
                        compression_token_embeddings.to(token_embeddings.device).to(token_embeddings.dtype),
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

                if prev_convergence is not None:
                    if init_method == "pretrained_pca":
                        pca_coefficients.grad[prev_convergence] = 0
                    else:
                        compression_token_embeddings.grad[prev_convergence] = 0

                if init_method == "pretrained_pca":
                    pca_coefficients_clone = pca_coefficients.detach().clone()
                else:
                    compression_token_embeddings_clone = compression_token_embeddings.detach().clone()

                optimizer.step()

                if prev_convergence is not None:
                    with torch.no_grad():
                        if init_method == "pretrained_pca":
                            pca_coefficients[prev_convergence] = pca_coefficients_clone[prev_convergence]
                        else:
                            compression_token_embeddings[prev_convergence] = compression_token_embeddings_clone[
                                prev_convergence
                            ]

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

                total_per_sample_convergence[step_i, :] = convergence_per_sample < 1.0
                total_per_sample_convergence_099[step_i, :] = convergence_per_sample < 0.99
                total_per_sample_convergence_095[step_i, :] = convergence_per_sample < 0.95
                prev_convergence = convergence_per_sample == 1.0

                if (convergence_per_sample == 1.0).all():
                    print(f"Early stopping: compression converged in {step_i} steps")
                    break

                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

            total_per_sample_convergence_sum = total_per_sample_convergence.sum(dim=0)
            print("total_per_sample_convergence_sum", total_per_sample_convergence_sum)
            total_per_sample_convergence_099_sum = total_per_sample_convergence_099.sum(dim=0)
            total_per_sample_convergence_095_sum = total_per_sample_convergence_095.sum(dim=0)

            with torch.no_grad():
                tokenizer = self.processing_class
                last_loss = loss.item()
                last_convergence_per_sample = convergence_per_sample.cpu()
                pca_coefficients_to_save = None
                if init_method == "pretrained_pca":
                    reconstructed_flat = torch.matmul(pca_coefficients, pca_components_device) + pca_mean_device.unsqueeze(0)
                    compression_token_embeddings_cpu = (
                        reconstructed_flat.reshape(batch_size, num_compression_tokens, hidden_size).detach().cpu()
                    )
                    pca_coefficients_to_save = pca_coefficients.clone().detach().to(torch.float32).cpu().numpy().tolist()
                else:
                    compression_token_embeddings_cpu = compression_token_embeddings.detach().cpu()
                if init_method == "pretrained_pca":
                    final_compression_tokens_for_ig = reconstructed_flat.reshape(
                        batch_size, num_compression_tokens, hidden_size
                    )
                else:
                    final_compression_tokens_for_ig = compression_token_embeddings

                per_sample_info_gain = []
                for j in range(batch_size):
                    sample_input_ids = input_ids[j : j + 1]
                    sample_attention_mask = attention_mask[j : j + 1]

                    sample_outputs_lm = model(input_ids=sample_input_ids, attention_mask=sample_attention_mask)
                    sample_logits_lm = sample_outputs_lm.logits
                    sample_shift_logits_lm = sample_logits_lm[:, :-1, :].contiguous()
                    sample_shift_labels_lm = sample_input_ids[:, 1:].contiguous()
                    sample_shift_mask_lm = sample_attention_mask[:, 1:].contiguous()

                    sample_shift_logits_lm_flat = sample_shift_logits_lm.view(-1, sample_shift_logits_lm.size(-1))
                    sample_shift_labels_lm_flat = sample_shift_labels_lm.view(-1)
                    sample_shift_mask_lm_flat = sample_shift_mask_lm.view(-1)

                    sample_valid_mask_lm = sample_shift_mask_lm_flat.bool()
                    if sample_valid_mask_lm.sum() > 0:
                        sample_ce_lm_sum = F.cross_entropy(
                            sample_shift_logits_lm_flat[sample_valid_mask_lm],
                            sample_shift_labels_lm_flat[sample_valid_mask_lm],
                            reduction="sum",
                        )
                        sample_H_LM_bits = sample_ce_lm_sum.item() / math.log(2)
                    else:
                        sample_H_LM_bits = 0.0

                    sample_inputs_embeds = token_embeddings[j : j + 1]
                    sample_compression_tokens = final_compression_tokens_for_ig[j : j + 1]
                    sample_model_tokens_with_compression = torch.cat(
                        [
                            sample_compression_tokens.to(sample_inputs_embeds.device).to(sample_inputs_embeds.dtype),
                            sample_inputs_embeds,
                        ],
                        dim=1,
                    )
                    sample_compression_attention_mask = compression_attention_mask[j : j + 1]
                    sample_attention_mask_with_compression = torch.cat(
                        [sample_compression_attention_mask, sample_attention_mask], dim=1
                    )

                    sample_outputs_mem = model(
                        inputs_embeds=sample_model_tokens_with_compression,
                        attention_mask=sample_attention_mask_with_compression,
                    )
                    sample_logits_mem = sample_outputs_mem.logits
                    sample_aligned_logits_mem = sample_logits_mem[:, num_compression_tokens:, :]
                    sample_shift_logits_mem = sample_aligned_logits_mem[:, :-1, :].contiguous()
                    sample_shift_labels_mem = sample_input_ids[:, 1:].contiguous()
                    sample_shift_mask_mem = sample_attention_mask[:, 1:].contiguous()

                    sample_shift_logits_mem_flat = sample_shift_logits_mem.view(-1, sample_shift_logits_mem.size(-1))
                    sample_shift_labels_mem_flat = sample_shift_labels_mem.view(-1)
                    sample_shift_mask_mem_flat = sample_shift_mask_mem.view(-1)

                    sample_valid_mask_mem = sample_shift_mask_mem_flat.bool()
                    if sample_valid_mask_mem.sum() > 0:
                        sample_ce_mem_sum = F.cross_entropy(
                            sample_shift_logits_mem_flat[sample_valid_mask_mem],
                            sample_shift_labels_mem_flat[sample_valid_mask_mem],
                            reduction="sum",
                        )
                        sample_H_LM_mem_bits = sample_ce_mem_sum.item() / math.log(2)
                    else:
                        sample_H_LM_mem_bits = 0.0

                    sample_info_gain = sample_H_LM_bits - sample_H_LM_mem_bits
                    per_sample_info_gain.append(sample_info_gain)
                    print("sample_info_gain", sample_info_gain)

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
                            "information_gain_bits": float(per_sample_info_gain[j]),
                        }
                    )
                    sample_id_counter += 1
                    final_compression_token_embeddings_cpu = compression_token_embeddings_cpu

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        save_path = self._save_artifacts(final_compression_token_embeddings_cpu, collected_rows, "compressed_prefixes")
        if save_path is not None:
            return save_path
        return None
