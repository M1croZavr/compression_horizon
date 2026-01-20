import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.decomposition import PCA
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import get_scheduler

from compression_horizon.inference.generation import generate_from_compression
from compression_horizon.train.loss import (
    compute_hybrid_cross_entropy_and_alignment_loss,
    token_argmax_match_rate_with_prefix,
)
from compression_horizon.utils.launch import (
    freeze_model_parameters,
    get_device,
    set_launch_seed,
)


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
        # TensorBoard
        log_dir = self.args.logging_dir
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None
        self.global_step = 0

    def compute_loss(
        self,
        model,
        input_ids,
        token_embeddings,
        attention_mask,
        united_token_embeddings,
        united_attention_mask,
        num_compression_tokens,
        target_hidden=None,
    ):
        loss_type = self.args.loss_type.lower()

        if loss_type != "cross_entropy":
            assert target_hidden is not None

        # Hidden state: [batch, mem + sequence, hidden]
        extra_kwargs = {}
        if self.args.fix_position_ids:
            position_ids = torch.arange(-1, token_embeddings.shape[1], device=token_embeddings.device)
            position_ids[0] = 0
            position_ids = position_ids.unsqueeze(0)
            # print('position_ids', position_ids)
            extra_kwargs["position_ids"] = position_ids

        compression_outputs = model(
            inputs_embeds=united_token_embeddings,
            attention_mask=united_attention_mask,
            output_hidden_states=(loss_type != "cross_entropy"),
            **extra_kwargs,
        )

        # Activation alignment loss
        hybrid_alpha = self.args.hybrid_alpha
        loss, alignment_loss = compute_hybrid_cross_entropy_and_alignment_loss(
            logits=compression_outputs.logits,
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_prefix_tokens=num_compression_tokens,
            target_hidden_states=target_hidden,
            compression_hidden_states=compression_outputs.hidden_states,
            num_alignment_layers=self.args.num_alignment_layers,
            inverted_alignment=self.args.inverted_alignment,
            loss_type=loss_type,
            hybrid_alpha=hybrid_alpha,
        )

        model.eval()
        with torch.no_grad():
            # Accuracy by logits
            convergence_per_sample = token_argmax_match_rate_with_prefix(
                compression_outputs.logits,
                input_ids,
                attention_mask,
                num_compression_tokens,
            )

            # Accuracy by autoregressive generation
            # Generate tokens from compressed trained embedding
            if self.global_step % 100 == 0 and self.args.generate_in_compute_loss:
                generated_text: Optional[list] = generate_from_compression(
                    model,
                    self.processing_class,
                    united_token_embeddings[:, :num_compression_tokens],
                    max_new_tokens=self.args.max_sequence_length,
                    num_return_sequences=1,
                )
                ground_truth_text: Optional[list] = self.processing_class.batch_decode(input_ids, skip_special_tokens=True)
            else:
                generated_text = None
                ground_truth_text = None
        model.eval()

        return (
            loss,
            alignment_loss,
            convergence_per_sample,
            generated_text,
            ground_truth_text,
        )

    def _prepare_embedding_init(self, model):
        init_method = self.args.embedding_init_method
        mvn_dist = None
        pca_components = None
        pca_mean = None
        loaded_embeddings = None

        if init_method == "load_from_disk":
            # Load embeddings from disk or generate if path is empty
            if not self.args.embedding_init_path or not os.path.exists(self.args.embedding_init_path):
                # Generate embeddings using the specified method
                if not self.args.embedding_init_path:
                    # Determine save path - use output_dir if available, otherwise current directory
                    if self.args.output_dir:
                        os.makedirs(self.args.output_dir, exist_ok=True)
                        save_path = os.path.join(self.args.output_dir, "generated_compression_embeddings.pt")
                    else:
                        save_path = "generated_compression_embeddings.pt"
                else:
                    # Path specified but doesn't exist - generate and save to that path
                    save_path = self.args.embedding_init_path
                    save_dir = os.path.dirname(save_path)
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)

                # Get model dimensions for generating embeddings
                hidden_size = model.config.hidden_size
                num_compression_tokens = self.args.number_of_mem_tokens

                # Prepare initialization for the generation method
                gen_init_method = self.args.load_from_disk_embedding_init_method
                gen_mvn_dist = None
                gen_pca_components = None
                gen_pca_mean = None
                gen_loaded_embeddings = None

                # Prepare initialization parameters for generation method
                if gen_init_method == "mvnormal":
                    with torch.no_grad():
                        emb_weight = None
                        try:
                            emb_weight = model.get_input_embeddings().weight
                        except Exception:
                            sd = model.state_dict()
                            if "transformer.wte.weight" in sd:
                                emb_weight = sd["transformer.wte.weight"]
                            else:
                                for k in sd.keys():
                                    if k.endswith("embed_tokens.weight") or k.endswith("wte.weight"):
                                        emb_weight = sd[k]
                                        break
                        if emb_weight is not None:
                            # Move to CPU for consistency
                            pre_expansion_embeddings = (emb_weight[:-3, :] if emb_weight.shape[0] > 3 else emb_weight).cpu()
                            mvn_mu = pre_expansion_embeddings.mean(dim=0).to(torch.float32)
                            n = pre_expansion_embeddings.size(0)
                            centered = pre_expansion_embeddings.to(torch.float32) - mvn_mu
                            sigma = (centered.T @ centered) / max(n, 1)
                            eps = 1e-6
                            sigma = sigma + eps * torch.eye(sigma.shape[0], device=sigma.device, dtype=sigma.dtype)
                            covariance = 1e-5 * sigma
                            try:
                                gen_mvn_dist = torch.distributions.MultivariateNormal(mvn_mu, covariance_matrix=covariance)
                            except Exception:
                                diag_cov = torch.clamp(torch.diag(covariance), min=1e-8)
                                gen_mvn_dist = torch.distributions.MultivariateNormal(
                                    mvn_mu, covariance_matrix=torch.diag(diag_cov)
                                )
                        else:
                            raise ValueError("cant run mv normal initialization method")
                elif gen_init_method == "pretrained_pca":
                    if not self.args.pretrained_pca_path:
                        raise ValueError(
                            "pretrained_pca_path must be specified when using load_from_disk_embedding_init_method=pretrained_pca"
                        )
                    if not os.path.exists(self.args.pretrained_pca_path):
                        raise ValueError(f"pretrained_pca_path does not exist: {self.args.pretrained_pca_path}")
                    progressive_ds = Dataset.load_from_disk(self.args.pretrained_pca_path)
                    all_embeddings = []
                    for i in range(len(progressive_ds)):
                        row = progressive_ds[i]
                        if int(row.get("sample_id", -1)) == 0:
                            embedding = row.get("embedding")
                            if embedding is not None:
                                if isinstance(embedding, list):
                                    emb_tensor = torch.tensor(embedding, dtype=torch.float32)
                                else:
                                    emb_tensor = torch.tensor(embedding, dtype=torch.float32)
                                emb_flat = emb_tensor.reshape(-1).to(torch.float32).detach().cpu().numpy()
                                all_embeddings.append(emb_flat)
                    if len(all_embeddings) == 0:
                        raise ValueError(f"No embeddings found for sample_id=0 in {self.args.pretrained_pca_path}")
                    X = np.stack(all_embeddings, axis=0)
                    n_components = min(self.args.pretrained_pca_num_components, X.shape[0] - 1, X.shape[1])
                    if n_components < 1:
                        raise ValueError(f"Cannot fit PCA: need at least 2 samples, got {X.shape[0]}")
                    pca = PCA(n_components=n_components, random_state=42)
                    pca.fit(X)
                    gen_pca_components = torch.tensor(pca.components_, dtype=torch.float32)
                    gen_pca_mean = torch.tensor(pca.mean_, dtype=torch.float32)

                # Generate embeddings using the specified method (batch_size=1, will be repeated later)
                generated_embeddings = self._init_compression_tokens(
                    1,
                    num_compression_tokens,
                    hidden_size,
                    gen_init_method,
                    gen_mvn_dist,
                    token_embeddings=None,
                    single_compressed_embeddings_initialization=None,
                    pca_components=gen_pca_components,
                    pca_mean=gen_pca_mean,
                    loaded_embeddings=gen_loaded_embeddings,
                )
                # Extract the actual tensor (remove Parameter wrapper) and save
                generated_embeddings_tensor = generated_embeddings.data.detach().clone().cpu()
                torch.save(generated_embeddings_tensor, save_path)
                print(
                    f"Generated embeddings using method '{gen_init_method}' and saved to {save_path}: shape {generated_embeddings_tensor.shape}"
                )
                loaded_embeddings = generated_embeddings_tensor
            else:
                # Load embeddings from existing file
                loaded_embeddings = torch.load(self.args.embedding_init_path, map_location="cpu")
                # Ensure it's a tensor and convert to float32
                if isinstance(loaded_embeddings, dict):
                    # If it's a dict, try common keys
                    if "compression_embeddings" in loaded_embeddings:
                        loaded_embeddings = loaded_embeddings["compression_embeddings"]
                    elif "state_dict" in loaded_embeddings:
                        # If state_dict, try to find embedding key
                        for key in loaded_embeddings["state_dict"].keys():
                            if "compression" in key.lower() or "embedding" in key.lower():
                                loaded_embeddings = loaded_embeddings["state_dict"][key]
                                break
                        else:
                            raise ValueError(
                                f"Could not find compression embeddings in state_dict at {self.args.embedding_init_path}"
                            )
                    else:
                        # Try first value
                        loaded_embeddings = next(iter(loaded_embeddings.values()))
                if not isinstance(loaded_embeddings, torch.Tensor):
                    loaded_embeddings = torch.tensor(loaded_embeddings, dtype=torch.float32)
                loaded_embeddings = loaded_embeddings.to(torch.float32)
                print(f"Loaded embeddings from {self.args.embedding_init_path}: shape {loaded_embeddings.shape}")

        elif init_method == "pretrained_pca":
            # Load PCA components from pretrained progressive dataset
            if not self.args.pretrained_pca_path:
                raise ValueError("pretrained_pca_path must be specified when using embedding_init_method=pretrained_pca")
            if not os.path.exists(self.args.pretrained_pca_path):
                raise ValueError(f"pretrained_pca_path does not exist: {self.args.pretrained_pca_path}")

            # Load progressive dataset
            progressive_ds = Dataset.load_from_disk(self.args.pretrained_pca_path)

            # Get first sample's embeddings across all stages
            all_embeddings = []
            for i in range(len(progressive_ds)):
                row = progressive_ds[i]
                embedding = row.get("embedding")
                if embedding is not None:
                    # Convert to numpy array and flatten if needed
                    if isinstance(embedding, list):
                        emb_tensor = torch.tensor(embedding, dtype=torch.float32)
                    else:
                        emb_tensor = torch.tensor(embedding, dtype=torch.float32)
                    # Flatten: [num_compression_tokens, hidden_size] -> [num_compression_tokens * hidden_size]
                    emb_flat = emb_tensor.reshape(-1).to(torch.float32).detach().cpu().numpy()
                    all_embeddings.append(emb_flat)

            if len(all_embeddings) == 0:
                raise ValueError(f"No embeddings found for sample_id=0 in {self.args.pretrained_pca_path}")

            # Stack embeddings: [n_stages, flattened_dim]
            X = np.stack(all_embeddings, axis=0)

            # Fit PCA
            n_components = min(self.args.pretrained_pca_num_components, X.shape[0] - 1, X.shape[1])
            if n_components < 1:
                raise ValueError(f"Cannot fit PCA: need at least 2 samples, got {X.shape[0]}")

            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(X)

            # Store PCA components and mean for later use
            pca_components = torch.tensor(pca.components_, dtype=torch.float32)  # [n_components, flattened_dim]
            pca_mean = torch.tensor(pca.mean_, dtype=torch.float32)  # [flattened_dim]
            print(
                f"Loaded PCA from {self.args.pretrained_pca_path}: {n_components} components, "
                f"explained variance: {pca.explained_variance_ratio_.sum():.4f}"
            )

        elif init_method == "mvnormal":
            with torch.no_grad():
                emb_weight = None
                try:
                    emb_weight = model.get_input_embeddings().weight
                except Exception:
                    sd = model.state_dict()
                    if "transformer.wte.weight" in sd:
                        emb_weight = sd["transformer.wte.weight"]
                    else:
                        for k in sd.keys():
                            if k.endswith("embed_tokens.weight") or k.endswith("wte.weight"):
                                emb_weight = sd[k]
                                break
                if emb_weight is not None:
                    pre_expansion_embeddings = emb_weight[:-3, :] if emb_weight.shape[0] > 3 else emb_weight
                    mvn_mu = pre_expansion_embeddings.mean(dim=0)
                    n = pre_expansion_embeddings.size(0)
                    centered = pre_expansion_embeddings - mvn_mu
                    sigma = (centered.T @ centered) / max(n, 1)
                    eps = 1e-6
                    sigma = sigma + eps * torch.eye(sigma.shape[0], device=sigma.device, dtype=sigma.dtype)
                    covariance = 1e-5 * sigma
                    try:
                        mvn_dist = torch.distributions.MultivariateNormal(
                            mvn_mu.to(torch.float32), covariance_matrix=covariance.to(torch.float32)
                        )
                    except Exception:
                        diag_cov = torch.clamp(torch.diag(covariance), min=1e-8)
                        mvn_dist = torch.distributions.MultivariateNormal(
                            mvn_mu.to(torch.float32), covariance_matrix=torch.diag(diag_cov).to(torch.float32)
                        )
                else:
                    raise ValueError("cant run mv normal initialization method")
        return init_method, mvn_dist, pca_components, pca_mean, loaded_embeddings

    def _create_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

    @staticmethod
    def _init_compression_tokens(
        batch_size,
        num_tokens,
        hidden_size,
        init_method,
        mvn_dist,
        token_embeddings=None,
        single_compressed_embeddings_initialization=None,
        pca_components=None,
        pca_mean=None,
        loaded_embeddings=None,
    ):
        if init_method == "mvnormal" and mvn_dist is not None:
            samples = mvn_dist.sample((batch_size, num_tokens))
            trainable_embeddings = torch.nn.Parameter(samples.to(dtype=torch.float32))
        elif init_method == "zeros":
            trainable_embeddings = torch.nn.Parameter(torch.zeros([batch_size, num_tokens, hidden_size], dtype=torch.float32))
        elif init_method == "single_random":
            if single_compressed_embeddings_initialization is not None:
                trainable_embeddings = torch.nn.Parameter(
                    single_compressed_embeddings_initialization.detach().clone().repeat(batch_size, 1, 1)
                )
            else:
                single_random_embedding = torch.rand([1, num_tokens, hidden_size], dtype=torch.float32)
                # assert batch_size == 1
                single_random_embedding = single_random_embedding.repeat(batch_size, 1, 1)
                trainable_embeddings = torch.nn.Parameter(single_random_embedding)
        elif init_method == "single_random0.02":
            if single_compressed_embeddings_initialization is not None:
                trainable_embeddings = torch.nn.Parameter(
                    single_compressed_embeddings_initialization.detach().clone().repeat(batch_size, 1, 1)
                )
            else:
                single_random_embedding = torch.rand([1, num_tokens, hidden_size], dtype=torch.float32)
                # assert batch_size == 1
                single_random_embedding = single_random_embedding.repeat(batch_size, 1, 1)
                trainable_embeddings = torch.nn.Parameter(single_random_embedding)
        elif init_method == "random":
            trainable_embeddings = torch.nn.Parameter(torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32))
        elif init_method == "random0.2":
            trainable_embeddings = torch.nn.Parameter(
                torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 0.2
            )
        elif init_method == "random0.02":
            trainable_embeddings = torch.nn.Parameter(
                torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 0.02
            )
        elif init_method == "random_norm":
            trainable_embeddings = torch.nn.Parameter(torch.randn([batch_size, num_tokens, hidden_size], dtype=torch.float32))
        elif init_method == "random_norm_0.2":
            trainable_embeddings = torch.nn.Parameter(
                torch.randn([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 0.2
            )
        elif init_method == "random_norm_0.02":
            trainable_embeddings = torch.nn.Parameter(
                torch.randn([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 0.02
            )
        elif init_method == "random0.002":
            trainable_embeddings = torch.nn.Parameter(
                torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 0.002
            )
        elif init_method == "random0.0002":
            trainable_embeddings = torch.nn.Parameter(
                torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 0.0002
            )
        elif init_method == "random5":
            trainable_embeddings = torch.nn.Parameter(
                torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 5
            )
        elif init_method == "neg_random":
            trainable_embeddings = torch.nn.Parameter(
                torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 2 - 1
            )
        elif init_method == "neg_random0.2":
            trainable_embeddings = torch.nn.Parameter(
                (torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 2 - 1) * 0.2
            )
        elif init_method == "neg_random5":
            trainable_embeddings = torch.nn.Parameter(
                (torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 2 - 1) * 5
            )
        elif init_method == "mean_token_embeds":
            assert token_embeddings is not None, "token_embeddings is required for `mean_token_embeds` init method"
            trainable_embeddings = torch.nn.Parameter(token_embeddings.mean(1, keepdim=True).repeat(1, num_tokens, 1))
        elif init_method == "pretrained_pca":
            assert pca_components is not None, "pca_components is required for `pretrained_pca` init method"
            assert pca_mean is not None, "pca_mean is required for `pretrained_pca` init method"
            # pca_components: [n_components, flattened_dim]
            # pca_mean: [flattened_dim]
            # flattened_dim = num_tokens * hidden_size (from the pretrained dataset)

            # Check if dimensions match
            flattened_dim = pca_mean.shape[0]
            expected_flattened_dim = num_tokens * hidden_size
            if flattened_dim != expected_flattened_dim:
                raise ValueError(
                    f"PCA dimension mismatch: pretrained has {flattened_dim} (num_tokens * hidden_size), "
                    f"but current needs {expected_flattened_dim} (num_tokens={num_tokens}, hidden_size={hidden_size})"
                )

            # Use PCA components to initialize: sample random coefficients in PCA space
            n_components_to_use = min(pca_components.shape[0], num_tokens)
            # Sample random coefficients: [batch_size, n_components_to_use]
            pca_coeffs = torch.randn([batch_size, n_components_to_use], dtype=torch.float32) * 0.1
            # Reconstruct: [batch, n_components] @ [n_components, flattened_dim] -> [batch, flattened_dim]
            reconstructed_flat = torch.matmul(pca_coeffs, pca_components[:n_components_to_use])  # [batch, flattened_dim]
            # Add mean
            reconstructed_flat = reconstructed_flat + pca_mean.unsqueeze(0)  # [batch, flattened_dim]
            # Reshape to [batch, num_tokens, hidden_size]
            trainable_embeddings = torch.nn.Parameter(reconstructed_flat.reshape(batch_size, num_tokens, hidden_size))
        elif init_method == "load_from_disk":
            assert loaded_embeddings is not None, "loaded_embeddings is required for `load_from_disk` init method"
            # Ensure loaded_embeddings has the correct shape
            # Expected shape: [num_tokens, hidden_size] or [1, num_tokens, hidden_size] or [batch_size, num_tokens, hidden_size]
            if len(loaded_embeddings.shape) == 2:
                # [num_tokens, hidden_size] -> repeat for batch
                if loaded_embeddings.shape[0] != num_tokens or loaded_embeddings.shape[1] != hidden_size:
                    raise ValueError(
                        f"Loaded embeddings shape mismatch: got {loaded_embeddings.shape}, "
                        f"expected [{num_tokens}, {hidden_size}] or [1, {num_tokens}, {hidden_size}]"
                    )
                trainable_embeddings = torch.nn.Parameter(
                    loaded_embeddings.unsqueeze(0).repeat(batch_size, 1, 1).to(torch.float32)
                )
            elif len(loaded_embeddings.shape) == 3:
                # [batch_or_1, num_tokens, hidden_size]
                if loaded_embeddings.shape[1] != num_tokens or loaded_embeddings.shape[2] != hidden_size:
                    raise ValueError(
                        f"Loaded embeddings shape mismatch: got {loaded_embeddings.shape}, "
                        f"expected [1, {num_tokens}, {hidden_size}] or [{batch_size}, {num_tokens}, {hidden_size}]"
                    )
                if loaded_embeddings.shape[0] == 1:
                    # Single embedding, repeat for batch
                    trainable_embeddings = torch.nn.Parameter(loaded_embeddings.repeat(batch_size, 1, 1).to(torch.float32))
                elif loaded_embeddings.shape[0] == batch_size:
                    # Already has correct batch size
                    trainable_embeddings = torch.nn.Parameter(loaded_embeddings.to(torch.float32))
                else:
                    raise ValueError(
                        f"Loaded embeddings batch size mismatch: got {loaded_embeddings.shape[0]}, "
                        f"expected 1 or {batch_size}"
                    )
            else:
                raise ValueError(f"Loaded embeddings must be 2D or 3D tensor, got shape {loaded_embeddings.shape}")
        else:
            raise ValueError(f"unsupported init method: {init_method}")
        return trainable_embeddings

    def _build_optimizer_and_scheduler(self, params, num_training_steps=None):
        if self.args.optim == "adamw_torch":
            optimizer = AdamW(
                params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
            )
        elif self.args.optim == "sgd":
            optimizer = SGD(
                params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError("Only SGD and adamw_torch are supported")

        lr_scheduler = None
        if num_training_steps is not None:
            print("self.args.lr_scheduler_type", self.args.lr_scheduler_type)
            scheduler_kwargs = {
                "optimizer": optimizer,
                "num_warmup_steps": self.args.warmup_steps,
                "num_training_steps": num_training_steps,
            }

            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type, **scheduler_kwargs, scheduler_specific_kwargs=self.args.lr_scheduler_kwargs
            )

        return optimizer, lr_scheduler

    def _log_step(
        self,
        loss,
        alignment_loss,
        convergence_per_sample,
        compression_token_embeddings,
        lr_scheduler,
        generated_text: Optional[list[str]],
        ground_truth_text: Optional[list[str]],
    ):
        if self.writer is None:
            return
        self.writer.add_scalar("train/loss", loss.item(), self.global_step)
        if alignment_loss is not None:
            self.writer.add_scalar("train/alignment_loss", alignment_loss.item(), self.global_step)
        self.writer.add_scalar("train/convergence", convergence_per_sample.mean().item(), self.global_step)
        self.writer.add_scalar(
            "compression_token_embeddings/mean",
            compression_token_embeddings.mean().item(),
            self.global_step,
        )
        self.writer.add_scalar(
            "compression_token_embeddings/std",
            compression_token_embeddings.std().item(),
            self.global_step,
        )
        grad_norm = compression_token_embeddings.grad.norm(2).item() if compression_token_embeddings.grad is not None else 0.0
        self.writer.add_scalar("train/grad_norm", grad_norm, self.global_step)
        if lr_scheduler is not None:
            lr_val = lr_scheduler.get_last_lr()[0]
            self.writer.add_scalar("train/lr", lr_val, self.global_step)
        if generated_text:
            self.writer.add_text("train/generated_text", " | ".join(generated_text), self.global_step)
        if ground_truth_text:
            self.writer.add_text(
                "train/ground_truth_text",
                " | ".join(ground_truth_text),
                self.global_step,
            )
        flush_steps = getattr(self.args, "logging_flush_steps", 100)
        if flush_steps and self.global_step % flush_steps == 0:
            self.writer.flush()
        self.global_step += 1

    def _save_artifacts(self, compression_token_embeddings: torch.Tensor, rows, subdir_name):
        output_dir = self.args.output_dir
        if output_dir and len(rows) > 0:
            os.makedirs(output_dir, exist_ok=True)
            if compression_token_embeddings is not None:
                save_path = os.path.join(output_dir, "compression_embeddings.pt")
                torch.save(compression_token_embeddings, save_path)
            save_path = os.path.join(output_dir, subdir_name)
            ds = Dataset.from_list(rows)
            ds.save_to_disk(save_path)
            return save_path
        return None

    def compute_target_hidden(self, model, token_embeddings, attention_mask):
        with torch.no_grad():
            # Hidden state: [batch, sequence, hidden]
            outputs = model(
                inputs_embeds=token_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            target_hidden = outputs.hidden_states
        return target_hidden

    def train(self):
        set_launch_seed(self.args.random_seed)
        device = get_device()
        model = self.model.to(device)
        freeze_model_parameters(model)
        init_method, mvn_dist, pca_components, pca_mean, loaded_embeddings = self._prepare_embedding_init(model)
        num_compression_tokens = self.args.number_of_mem_tokens

        # Collect per-sample artifacts for optional saving
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
            single_compressed_embeddings_initialization = (
                single_compressed_embeddings_initialization.data.detach().clone()
            )  # [batch, mem, hidden]

        dataloader = self._create_dataloader()
        for batch in tqdm(dataloader):
            model.eval()
            input_ids = batch.input_ids.squeeze(1)  # [batch, sequence]
            # print("input_ids", input_ids.shape)
            batch_size = input_ids.shape[0]

            attention_mask = batch.attention_mask.squeeze(1)  # [batch, sequence]
            with torch.no_grad():
                token_embeddings = model.get_input_embeddings()(input_ids)  # [batch, sequence, hidden]

            target_hidden = self.compute_target_hidden(model, token_embeddings, attention_mask)

            # Handle pretrained_pca initialization: optimize only coefficients
            if init_method == "pretrained_pca":
                assert pca_components is not None, "pca_components is required for pretrained_pca"
                assert pca_mean is not None, "pca_mean is required for pretrained_pca"

                # Move PCA components and mean to device
                pca_components_device = pca_components.to(device)  # [n_components, flattened_dim]
                pca_mean_device = pca_mean.to(device)  # [flattened_dim]

                # Validate dimensions
                flattened_dim = pca_mean_device.shape[0]
                expected_flattened_dim = num_compression_tokens * hidden_size
                if flattened_dim != expected_flattened_dim:
                    raise ValueError(
                        f"PCA dimension mismatch: pretrained has {flattened_dim}, "
                        f"but current needs {expected_flattened_dim} (num_tokens={num_compression_tokens}, hidden_size={hidden_size})"
                    )

                # Initialize coefficients: [batch_size, n_components]
                n_components = pca_components_device.shape[0]
                pca_coefficients = torch.nn.Parameter(
                    torch.randn([batch_size, n_components], dtype=torch.float32, device=device) * 0.1
                )

                # Reconstruct initial compression tokens for saving initialization
                reconstructed_flat = torch.matmul(pca_coefficients, pca_components_device) + pca_mean_device.unsqueeze(0)
                initialization_embeddings = (
                    reconstructed_flat.reshape(batch_size, num_compression_tokens, hidden_size).detach().cpu()
                )

                # Optimizer only optimizes coefficients
                optimizer, lr_scheduler = self._build_optimizer_and_scheduler(
                    [pca_coefficients], num_training_steps=self.args.max_optimization_steps_per_sample
                )
            else:
                # Standard initialization: optimize full compression tokens
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
                )  # [batch, mem, hidden]
                # Move to device and save initialization embedding (before optimization)
                # Create new Parameter on device to avoid non-leaf tensor issue
                compression_token_embeddings = torch.nn.Parameter(compression_token_embeddings.data.to(device))
                initialization_embeddings = compression_token_embeddings.detach().clone().cpu()  # [batch, mem, hidden]
                optimizer, lr_scheduler = self._build_optimizer_and_scheduler(
                    [compression_token_embeddings], num_training_steps=self.args.max_optimization_steps_per_sample
                )

            compression_attention_mask = torch.tensor([1], dtype=attention_mask.dtype).repeat(
                batch_size, num_compression_tokens
            )  # [batch, mem]

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
                # disable=True,
            )
            progress_bar.set_description("Training")

            total_per_sample_convergence = torch.zeros(
                [
                    self.args.max_optimization_steps_per_sample,
                    input_ids.shape[0],
                ],
                dtype=torch.long,
            )
            prev_convergence = None
            # prev_convergence_per_sample = None
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
                # Reconstruct compression tokens from PCA coefficients if using pretrained_pca
                if init_method == "pretrained_pca":
                    # Reconstruct: [batch, n_components] @ [n_components, flattened_dim] + [flattened_dim] -> [batch, flattened_dim]
                    reconstructed_flat = torch.matmul(pca_coefficients, pca_components_device) + pca_mean_device.unsqueeze(0)
                    compression_token_embeddings = reconstructed_flat.reshape(batch_size, num_compression_tokens, hidden_size)
                # else: compression_token_embeddings is already defined in the outer scope

                # Rebuild concatenations each step to avoid reusing the same autograd graph
                united_token_embeddings = torch.cat(
                    [compression_token_embeddings.to(token_embeddings.device).to(token_embeddings.dtype), token_embeddings],
                    dim=1,
                )  # [batch, mem + sequence, hidden]
                united_attention_mask = torch.cat(
                    [compression_attention_mask, attention_mask],
                    dim=1,
                )  # [batch, mem + sequence]
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
                # Calculate gradients and update compression embeddings
                loss.backward()

                if prev_convergence is not None:
                    # Zero gradients for converged items
                    if init_method == "pretrained_pca":
                        # Zero gradients for converged items' coefficients
                        pca_coefficients.grad[prev_convergence] = 0
                    else:
                        # Zero gradients for converged items' compression tokens
                        compression_token_embeddings.grad[prev_convergence] = 0
                    # print(
                    #     "Non zero gradients:",
                    #     (compression_token_embeddings.grad.sum(-1) != 0).sum(),
                    #     "/",
                    #     united_token_embeddings.shape[0],
                    #     "prev_convergence_per_sample",
                    #     prev_convergence_per_sample,
                    # )

                if init_method == "pretrained_pca":
                    pca_coefficients_clone = pca_coefficients.detach().clone()
                else:
                    compression_token_embeddings_clone = compression_token_embeddings.detach().clone()

                optimizer.step()

                if prev_convergence is not None:
                    with torch.no_grad():
                        if init_method == "pretrained_pca":
                            # Restore converged items' coefficients
                            pca_coefficients[prev_convergence] = pca_coefficients_clone[prev_convergence]
                        else:
                            # Restore converged items' compression tokens
                            compression_token_embeddings[prev_convergence] = compression_token_embeddings_clone[
                                prev_convergence
                            ]

                # Log current step progress
                with torch.no_grad():
                    progress_bar.update(1)
                    alignment_loss_item = None
                    if alignment_loss is not None:
                        alignment_loss_item = alignment_loss.item()
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
                # prev_convergence_per_sample = convergence_per_sample

                print("convergence_per_sample", convergence_per_sample, convergence_per_sample == 1.0)
                if (convergence_per_sample == 1.0).all():
                    print(f"Early stopping: compression converged in {step_i} steps")
                    break

                # Update learning rate
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

            total_per_sample_convergence_sum = total_per_sample_convergence.sum(dim=0)
            print("total_per_sample_convergence_sum", total_per_sample_convergence_sum)
            total_per_sample_convergence_099_sum = total_per_sample_convergence_099.sum(dim=0)
            print("total_per_sample_convergence_099_sum", total_per_sample_convergence_099_sum)
            total_per_sample_convergence_095_sum = total_per_sample_convergence_095.sum(dim=0)
            print("total_per_sample_convergence_095_sum", total_per_sample_convergence_095_sum)

            # After optimizing this batch's compression tokens, record artifacts per sample (once per sample)
            with torch.no_grad():
                tokenizer = self.processing_class
                last_loss = loss.item()
                last_convergence_per_sample = convergence_per_sample.cpu()
                # Reconstruct compression tokens if using PCA (for saving)
                pca_coefficients_to_save = None
                if init_method == "pretrained_pca":
                    reconstructed_flat = torch.matmul(pca_coefficients, pca_components_device) + pca_mean_device.unsqueeze(0)
                    compression_token_embeddings_cpu = (
                        reconstructed_flat.reshape(batch_size, num_compression_tokens, hidden_size).detach().cpu()
                    )
                    pca_coefficients_to_save = pca_coefficients.clone().detach().to(torch.float32).cpu().numpy().tolist()
                else:
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
                            "embedding": embedding,  # [mem, hidden]
                            "pca_coefficients": pca_coefficients_to_save[j] if pca_coefficients_to_save is not None else None,
                            "initialization_embedding": initialization_embedding,  # [mem, hidden] - state before optimization
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
                    # Store final compression tokens for saving (from last batch)
                    final_compression_token_embeddings_cpu = compression_token_embeddings_cpu

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        # Persist artifacts
        save_path = self._save_artifacts(final_compression_token_embeddings_cpu, collected_rows, "compressed_prefixes")
        if save_path is not None:
            return save_path
        return None

    def train_low_dim(self):

        print("Train low dim!!")

        set_launch_seed(self.args.random_seed)
        device = get_device()
        model = self.model.to(device)
        freeze_model_parameters(model)
        init_method, mvn_dist, pca_components, pca_mean, loaded_embeddings = self._prepare_embedding_init(model)
        num_compression_tokens = self.args.number_of_mem_tokens

        # Collect per-sample artifacts for optional saving
        collected_rows = []
        sample_id_counter = 0

        hidden_size = model.config.hidden_size

        dataloader = self._create_dataloader()
        final_projection = None
        for batch in tqdm(dataloader):
            model.eval()
            input_ids = batch.input_ids.squeeze(1)  # [batch, sequence]
            # print("input_ids", input_ids.shape)
            batch_size = input_ids.shape[0]

            attention_mask = batch.attention_mask.squeeze(1)  # [batch, sequence]
            with torch.no_grad():
                token_embeddings = model.get_input_embeddings()(input_ids)  # [batch, sequence, hidden]

            target_hidden = self.compute_target_hidden(model, token_embeddings, attention_mask)

            # Standard initialization: optimize full compression tokens
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
            )  # [batch, mem, low_dim_size]

            projection, projection_optimizer, projection_lr_scheduler = self._prepare_low_dim_proj(embedding_dim=hidden_size)
            projection = projection.to(device)
            print("projection_optimizer", projection_optimizer)

            # Move to device and save initialization embedding (before optimization)
            # Create new Parameter on device to avoid non-leaf tensor issue
            compression_token_embeddings = torch.nn.Parameter(compression_token_embeddings.data.to(device))
            initialization_embeddings = compression_token_embeddings.detach().clone().cpu()  # [batch, mem, hidden]
            optimizer, lr_scheduler = self._build_optimizer_and_scheduler(
                [compression_token_embeddings], num_training_steps=self.args.max_optimization_steps_per_sample
            )

            compression_attention_mask = torch.tensor([1], dtype=attention_mask.dtype).repeat(
                batch_size, num_compression_tokens
            )  # [batch, mem]

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
                # disable=True,
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
                # Reconstruct compression tokens from PCA coefficients if using pretrained_pca
                # Rebuild concatenations each step to avoid reusing the same autograd graph

                compression_token_embeddings_llm = projection(compression_token_embeddings)

                united_token_embeddings = torch.cat(
                    [compression_token_embeddings_llm.to(token_embeddings.device).to(token_embeddings.dtype), token_embeddings],
                    dim=1,
                )  # [batch, mem + sequence, hidden]
                united_attention_mask = torch.cat(
                    [compression_attention_mask, attention_mask],
                    dim=1,
                )  # [batch, mem + sequence]
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
                # Calculate gradients and update compression embeddings
                loss.backward()

                # compression_token_embeddings_clone = compression_token_embeddings.detach().clone()

                optimizer.step()
                if projection_optimizer is not None:
                    projection_optimizer.step()

                # Log current step progress
                with torch.no_grad():
                    progress_bar.update(1)
                    alignment_loss_item = None
                    if alignment_loss is not None:
                        alignment_loss_item = alignment_loss.item()
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

                    print("convergence_per_sample", convergence_per_sample, convergence_per_sample == 1.0)
                    if (convergence_per_sample == 1.0).all():
                        print(f"Early stopping: compression converged in {step_i} steps")
                        break

                total_per_sample_convergence[step_i, :] = convergence_per_sample < 1.0
                total_per_sample_convergence_099[step_i, :] = convergence_per_sample < 0.99
                total_per_sample_convergence_095[step_i, :] = convergence_per_sample < 0.95

                # Update learning rate
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                if projection_optimizer is not None:
                    projection_optimizer.zero_grad(set_to_none=True)
                if projection_lr_scheduler is not None:
                    projection_lr_scheduler.step()

            total_per_sample_convergence_sum = total_per_sample_convergence.sum(dim=0)
            print("total_per_sample_convergence_sum", total_per_sample_convergence_sum)
            total_per_sample_convergence_099_sum = total_per_sample_convergence_099.sum(dim=0)
            print("total_per_sample_convergence_099_sum", total_per_sample_convergence_099_sum)
            total_per_sample_convergence_095_sum = total_per_sample_convergence_095.sum(dim=0)
            print("total_per_sample_convergence_095_sum", total_per_sample_convergence_095_sum)

            # After optimizing this batch's compression tokens, record artifacts per sample (once per sample)
            with torch.no_grad():
                tokenizer = self.processing_class
                last_loss = loss.item()
                last_convergence_per_sample = convergence_per_sample.cpu()
                # Reconstruct compression tokens if using PCA (for saving)
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
                            "embedding": embedding,  # [mem, hidden]
                            "pca_coefficients": pca_coefficients_to_save[j] if pca_coefficients_to_save is not None else None,
                            "initialization_embedding": initialization_embedding,  # [mem, hidden] - state before optimization
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
                    # Store final compression tokens for saving (from last batch)
                    final_compression_token_embeddings_cpu = compression_token_embeddings_cpu

            # Track final projection for saving (from last batch)
            final_projection = projection

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        # Save projection weights if training was enabled
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

        # Persist artifacts
        save_path = self._save_artifacts(final_compression_token_embeddings_cpu, collected_rows, "compressed_prefixes")
        if save_path is not None:
            return save_path
        return None

    def _evaluate_noop_on_longer_sequences(self, model, compression_token_embeddings_single, num_compression_tokens):
        """Evaluate compression embeddings on sequences that are twice as long as training sequences."""
        model.eval()
        device = next(model.parameters()).device
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

        all_convergences = []
        eval_seq_length = None

        with torch.no_grad():
            compression_token_embeddings_single_eval = compression_token_embeddings_single.to(device)
            for batch in eval_dataloader:
                input_ids = batch.input_ids.squeeze(1)  # [batch, sequence]
                batch_size = input_ids.shape[0]
                attention_mask = batch.attention_mask.squeeze(1)  # [batch, sequence]

                if eval_seq_length is None:
                    eval_seq_length = input_ids.shape[1]

                compression_token_embeddings = compression_token_embeddings_single_eval.repeat([batch_size, 1, 1])
                token_embeddings = model.get_input_embeddings()(input_ids)  # [batch, sequence, hidden]

                compression_attention_mask = torch.tensor([1], dtype=attention_mask.dtype).repeat(
                    batch_size, num_compression_tokens
                )  # [batch, mem]

                united_token_embeddings = torch.cat(
                    [compression_token_embeddings.to(token_embeddings.device).to(token_embeddings.dtype), token_embeddings],
                    dim=1,
                )  # [batch, mem + sequence, hidden]
                united_attention_mask = torch.cat([compression_attention_mask, attention_mask], dim=1)

                # Get base model predictions
                base_logits = model(input_ids, attention_mask=attention_mask).logits
                # Get compression model predictions
                united_logits = model(inputs_embeds=united_token_embeddings, attention_mask=united_attention_mask).logits

                # Compute convergence: compare united_logits argmax with base_logits argmax
                base_preds = base_logits.argmax(dim=-1)  # [batch, sequence]
                united_preds = united_logits[:, num_compression_tokens:, :].argmax(dim=-1)  # [batch, sequence]
                convergence_numerator = (united_preds == base_preds).sum(dim=-1)
                convergence_per_sample = convergence_numerator / attention_mask.sum(dim=-1)

                all_convergences.extend(convergence_per_sample.cpu().numpy().tolist())

        mean_convergence = float(torch.mean(torch.tensor(all_convergences)).item())
        return {
            "mean_convergence": mean_convergence,
            "all_convergences": all_convergences,
            "eval_seq_length": eval_seq_length,
        }

    def train_noop(self):
        set_launch_seed(self.args.random_seed)
        device = get_device()
        model = self.model.to(device)
        freeze_model_parameters(model)
        init_method, mvn_dist, pca_components, pca_mean, loaded_embeddings = self._prepare_embedding_init(model)
        num_compression_tokens = self.args.number_of_mem_tokens

        # Collect per-sample artifacts for optional saving
        collected_rows = []
        sample_id_counter = 0

        hidden_size = model.config.hidden_size

        compression_token_embeddings_single = self._init_compression_tokens(
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
        # Move to device and save initialization embedding (before optimization) - shared across all samples in train_noop
        # Create new Parameter on device to avoid non-leaf tensor issue
        compression_token_embeddings_single = torch.nn.Parameter(compression_token_embeddings_single.data.to(device))
        initialization_embedding_single = compression_token_embeddings_single.detach().clone().cpu()  # [1, mem, hidden]

        dataloader = self._create_dataloader()
        num_training_steps = self.args.num_train_epochs * len(dataloader) / self.args.gradient_accumulation_steps
        optimizer, lr_scheduler = self._build_optimizer_and_scheduler(
            [compression_token_embeddings_single], num_training_steps=num_training_steps
        )

        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        accumulation_step = 0
        early_stopped = False

        pbar = tqdm(range(int(self.args.num_train_epochs)))
        for epoch_i in pbar:
            if early_stopped:
                break

            for batch in dataloader:
                model.eval()
                input_ids = batch.input_ids.squeeze(1)  # [batch, sequence]
                batch_size = input_ids.shape[0]

                compression_token_embeddings = compression_token_embeddings_single.repeat([batch_size, 1, 1])

                attention_mask = batch.attention_mask.squeeze(1)  # [batch, sequence]
                with torch.no_grad():
                    token_embeddings = model.get_input_embeddings()(input_ids)  # [batch, sequence, hidden]

                # Trainable compression tokens per sample
                compression_attention_mask = torch.tensor([1], dtype=attention_mask.dtype).repeat(
                    batch_size, num_compression_tokens
                )  # [batch, mem]

                united_token_embeddings = torch.cat(
                    [compression_token_embeddings.to(token_embeddings.dtype), token_embeddings],
                    dim=1,
                )  # [batch, mem + sequence, hidden]
                united_attention_mask = torch.cat(
                    [compression_attention_mask, attention_mask],
                    dim=1,
                )  # [batch, mem + sequence]

                base_logits = model(input_ids, attention_mask=attention_mask).logits
                united_logits = model(inputs_embeds=united_token_embeddings, attention_mask=united_attention_mask).logits

                # Create distribution-like target: set non-top-k elements to -inf, then softmax
                max_tokens = self.args.max_tokens_in_distribution
                target_logits = base_logits.clone()
                # Get top-k tokens for each position
                topk_values, topk_indices = torch.topk(base_logits, k=max_tokens, dim=-1)  # [batch, sequence, k]
                # Create mask for top-k positions
                batch_size, seq_len, vocab_size = base_logits.shape
                topk_mask = torch.zeros_like(target_logits, dtype=torch.bool)
                topk_mask.scatter_(2, topk_indices, True)
                # Set non-top-k elements to -inf
                target_logits[~topk_mask] = float("-inf")
                # Apply softmax to get distribution with top-k tokens
                target_distribution = F.softmax(target_logits, dim=-1)

                # Use KL divergence for distribution target
                united_logits_sliced = united_logits[:, num_compression_tokens:, :]  # [batch, sequence, vocab]
                united_log_probs = F.log_softmax(united_logits_sliced, dim=-1)
                # Mask out padding positions by creating a valid positions mask
                valid_mask = attention_mask.bool()  # [batch, sequence]
                # Flatten for loss computation
                united_log_probs_flat = united_log_probs.flatten(0, 1)  # [batch*sequence, vocab]
                target_distribution_flat = target_distribution.flatten(0, 1)  # [batch*sequence, vocab]
                valid_mask_flat = valid_mask.flatten()  # [batch*sequence]
                # Only compute loss on valid (non-padding) positions
                united_log_probs_valid = united_log_probs_flat[valid_mask_flat]  # [num_valid, vocab]
                target_distribution_valid = target_distribution_flat[valid_mask_flat]  # [num_valid, vocab]
                # Compute KL divergence: sum over vocab dimension, mean over valid positions
                loss = F.kl_div(united_log_probs_valid, target_distribution_valid, reduction="batchmean")
                # Scale loss by gradient accumulation steps to maintain effective learning rate
                loss = loss / gradient_accumulation_steps

                # Calculate gradients and accumulate
                loss.backward()
                accumulation_step += 1

                # Update weights only after accumulating enough gradients
                if accumulation_step >= gradient_accumulation_steps:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                    accumulation_step = 0

                    # Evaluation: compute convergence metrics
                    model.eval()
                    with torch.no_grad():
                        # Recompute logits for evaluation
                        base_logits_eval = model(input_ids, attention_mask=attention_mask).logits
                        united_logits_eval = model(
                            inputs_embeds=united_token_embeddings, attention_mask=united_attention_mask
                        ).logits

                        # Compute convergence: compare united_logits argmax with base_logits argmax
                        base_preds = base_logits_eval.argmax(dim=-1)  # [batch, sequence]
                        united_preds = united_logits_eval[:, num_compression_tokens:, :].argmax(dim=-1)  # [batch, sequence]
                        convergence_numerator = (united_preds == base_preds).sum(dim=-1)
                        convergence_per_sample = convergence_numerator / attention_mask.sum(dim=-1)

                        # Generate text periodically for evaluation
                        generated_text: Optional[list] = None
                        ground_truth_text: Optional[list] = None
                        if self.global_step % 100 == 0:
                            generated_text = generate_from_compression(
                                model,
                                self.processing_class,
                                united_token_embeddings[:, :num_compression_tokens],
                                max_new_tokens=self.args.max_sequence_length,
                                num_return_sequences=1,
                            )
                            ground_truth_text = self.processing_class.batch_decode(input_ids, skip_special_tokens=True)

                    model.eval()

                    # Log metrics
                    self._log_step(
                        loss * gradient_accumulation_steps,
                        None,  # alignment_loss not used in train_noop
                        convergence_per_sample,
                        compression_token_embeddings_single,
                        lr_scheduler,
                        generated_text,
                        ground_truth_text,
                    )

                    # Update progress bar
                    pbar.set_postfix(
                        loss=loss.item() * gradient_accumulation_steps,
                        convergence=convergence_per_sample.mean().item(),
                        lr=lr_scheduler.get_last_lr()[0],
                        comp_emb_mean=compression_token_embeddings_single.mean().item(),
                        comp_emb_std=compression_token_embeddings_single.std().item(),
                    )

                    # Early stopping: check if convergence threshold reached
                    convergence_threshold = self.args.noop_convergence_threshold
                    if (convergence_per_sample >= convergence_threshold).all():
                        print(
                            f"Early stopping: convergence reached threshold {convergence_threshold} at step {self.global_step}"
                        )
                        early_stopped = True
                        break
                if early_stopped:
                    break

        # Handle any remaining accumulated gradients at the end of training
        if accumulation_step > 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

        # Record artifacts for remaining batches
        with torch.no_grad():
            tokenizer = self.processing_class
            compression_token_embeddings_cpu = compression_token_embeddings_single.detach().cpu()
            # Get final evaluation metrics
            all_convergences = []
            model.eval()
            for batch in dataloader:
                input_ids = batch.input_ids.squeeze(1)
                batch_size = input_ids.shape[0]
                attention_mask = batch.attention_mask.squeeze(1)
                compression_token_embeddings = compression_token_embeddings_single.repeat([batch_size, 1, 1])
                with torch.no_grad():
                    token_embeddings = model.get_input_embeddings()(input_ids)
                compression_attention_mask = torch.tensor([1], dtype=attention_mask.dtype).repeat(
                    batch_size, num_compression_tokens
                )
                united_token_embeddings = torch.cat(
                    [compression_token_embeddings.to(token_embeddings.device).to(token_embeddings.dtype), token_embeddings],
                    dim=1,
                )
                united_attention_mask = torch.cat([compression_attention_mask, attention_mask], dim=1)
                base_logits_eval = model(input_ids, attention_mask=attention_mask).logits
                united_logits_eval = model(inputs_embeds=united_token_embeddings, attention_mask=united_attention_mask).logits
                base_preds = base_logits_eval.argmax(dim=-1)
                united_preds = united_logits_eval[:, num_compression_tokens:, :].argmax(dim=-1)
                convergence_numerator = (united_preds == base_preds).sum(dim=-1)
                convergence_per_sample = convergence_numerator / attention_mask.sum(dim=-1)
                all_convergences.extend(convergence_per_sample.cpu().numpy().tolist())

                for j in range(batch_size):
                    sample_attention_mask = attention_mask[j].bool()
                    sample_input_ids = input_ids[j][sample_attention_mask]
                    sample_text = tokenizer.decode(sample_input_ids, skip_special_tokens=True)
                    embedding = compression_token_embeddings_cpu.to(torch.float32).numpy().tolist()
                    initialization_embedding = initialization_embedding_single[0].to(torch.float32).numpy().tolist()
                    collected_rows.append(
                        {
                            "sample_id": sample_id_counter,
                            "text": sample_text,
                            "embedding": embedding,
                            "initialization_embedding": initialization_embedding,  # [mem, hidden] - state before optimization
                            "final_loss": None,  # Loss not computed in final eval
                            "final_convergence": convergence_per_sample[j].item(),
                            "compression_tokens_mean": float(compression_token_embeddings_cpu.mean().item()),
                            "compression_tokens_std": float(compression_token_embeddings_cpu.std().item()),
                            "num_input_tokens": int(sample_attention_mask.sum().item()),
                            "num_compression_tokens": int(num_compression_tokens),
                            "hidden_size": hidden_size,
                            "loss_type": "noop_kl_div",
                            "max_tokens_in_distribution": self.args.max_tokens_in_distribution,
                            "dtype": self.args.dtype,
                            "embedding_init_method": self.args.embedding_init_method,
                            "model_checkpoint": self.args.model_checkpoint,
                            "num_train_epochs": self.args.num_train_epochs,
                        }
                    )
                    sample_id_counter += 1

            print("all_convergences mean", torch.mean(torch.tensor(all_convergences)))

            model.eval()

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        # Evaluate on longer sequences if eval_dataset is provided
        if self.eval_dataset is not None:
            print("Evaluating compression embeddings on longer sequences...")
            eval_results = self._evaluate_noop_on_longer_sequences(
                model, compression_token_embeddings_single, num_compression_tokens
            )
            print(f"Evaluation on longer sequences - Mean convergence: {eval_results['mean_convergence']:.4f}")
            # Add evaluation results to collected_rows metadata
            for row in collected_rows:
                row["eval_longer_seq_mean_convergence"] = eval_results["mean_convergence"]
                row["eval_longer_seq_length"] = eval_results["eval_seq_length"]

        # Persist artifacts
        save_path = self._save_artifacts(compression_token_embeddings_single.detach().cpu(), collected_rows, "noop_prefixes")
        if save_path is not None:
            return save_path
        return None

    def _prepare_low_dim_proj(self, embedding_dim):
        low_dim_prjoection = nn.Linear(self.args.low_dim_size, embedding_dim)

        # Load checkpoint if specified
        if self.args.low_dim_proj_checkpoint is not None:
            if not os.path.exists(self.args.low_dim_proj_checkpoint):
                raise ValueError(f"low_dim_proj_checkpoint does not exist: {self.args.low_dim_proj_checkpoint}")
            checkpoint = torch.load(self.args.low_dim_proj_checkpoint, map_location="cpu")
            # Load projection state_dict
            if isinstance(checkpoint, dict):
                if "low_dim_projection" in checkpoint:
                    low_dim_prjoection.load_state_dict(checkpoint["low_dim_projection"])
                elif "state_dict" in checkpoint:
                    low_dim_prjoection.load_state_dict(checkpoint["state_dict"])
                else:
                    # Assume the checkpoint is the state_dict itself
                    low_dim_prjoection.load_state_dict(checkpoint)
            else:
                # Assume the checkpoint is the state_dict itself
                low_dim_prjoection.load_state_dict(checkpoint)
            print(
                f"Loaded low-dimensional projection state from {self.args.low_dim_proj_checkpoint}, low dim size = {self.args.low_dim_size}"
            )

        # Only create optimizer and scheduler if training is enabled
        if self.args.low_dim_proj_train:
            low_dim_optim = AdamW(
                low_dim_prjoection.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay
            )
            scheduler_kwargs = {
                "optimizer": low_dim_optim,
                "num_warmup_steps": self.args.warmup_steps,
                "num_training_steps": self.args.max_optimization_steps_per_sample,
            }

            low_dim_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                **scheduler_kwargs,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
        else:
            # Freeze the projection parameters
            for param in low_dim_prjoection.parameters():
                param.requires_grad = False
            low_dim_optim = None
            low_dim_scheduler = None

        return low_dim_prjoection, low_dim_optim, low_dim_scheduler

    def progressive_train(self):
        device = get_device()
        set_launch_seed(self.args.random_seed)

        model = self.model.to(device)
        freeze_model_parameters(model)
        init_method, mvn_dist, pca_components, pca_mean, loaded_embeddings = self._prepare_embedding_init(model)

        dataloader = self._create_dataloader()

        num_compression_tokens = self.args.number_of_mem_tokens
        threshold = self.args.progressive_convergence_threshold
        step_increment = self.args.progressive_step
        min_len = self.args.progressive_min_seq_len
        max_stages_cap = self.args.progressive_max_stages

        collected_rows = []
        sample_id_counter = 0

        # model = torch.compile(model, mode='reduce-overhead')

        low_dim_prjoection = None
        low_dim_optim = None
        if self.args.low_dim_projection and self.args.low_dim_projection_global:
            low_dim_prjoection, low_dim_optim, low_dim_scheduler = self._prepare_low_dim_proj(
                embedding_dim=model.get_input_embeddings().embedding_dim
            )

        for batch in tqdm(dataloader):
            batch_size = batch["input_ids"].shape[0]
            full_input_ids = batch.input_ids.squeeze(1)
            with torch.no_grad():
                full_model_token_embeddings = model.get_input_embeddings()(full_input_ids)
            full_attention_mask = batch.attention_mask.squeeze(1)

            target_hidden_full = self.compute_target_hidden(model, full_model_token_embeddings, full_attention_mask)

            hidden_size = full_model_token_embeddings.shape[-1]
            if self.args.low_dim_projection:
                hidden_size = self.args.low_dim_size

            device = full_model_token_embeddings.device

            if self.args.low_dim_projection and not self.args.low_dim_projection_global:
                low_dim_prjoection, low_dim_optim, low_dim_scheduler = self._prepare_low_dim_proj(
                    embedding_dim=model.get_input_embeddings().embedding_dim
                )
                print("low_dim_prjoection", low_dim_prjoection, "low_dim_optim", low_dim_optim)

            # Handle pretrained_pca initialization: optimize only coefficients
            if init_method == "pretrained_pca":
                assert pca_components is not None, "pca_components is required for pretrained_pca"
                assert pca_mean is not None, "pca_mean is required for pretrained_pca"

                # Move PCA components and mean to device
                pca_components_device = pca_components.to(device)  # [n_components, flattened_dim]
                pca_mean_device = pca_mean.to(device)  # [flattened_dim]

                # Validate dimensions
                flattened_dim = pca_mean_device.shape[0]
                expected_flattened_dim = num_compression_tokens * hidden_size
                if flattened_dim != expected_flattened_dim:
                    raise ValueError(
                        f"PCA dimension mismatch: pretrained has {flattened_dim}, "
                        f"but current needs {expected_flattened_dim} (num_tokens={num_compression_tokens}, hidden_size={hidden_size})"
                    )

                # Initialize coefficients: [batch_size, n_components]
                n_components = pca_components_device.shape[0]
                pca_coefficients = torch.nn.Parameter(
                    torch.randn([batch_size, n_components], dtype=torch.float32, device=device) * 0.1
                )

                # Reconstruct initial compression tokens for saving initialization
                reconstructed_flat = torch.matmul(pca_coefficients, pca_components_device) + pca_mean_device.unsqueeze(0)
                initialization_embeddings = (
                    reconstructed_flat.reshape(batch_size, num_compression_tokens, hidden_size).detach().cpu()
                )

                # Optimizer only optimizes coefficients
                optimizer, lr_scheduler = self._build_optimizer_and_scheduler([pca_coefficients])
            else:
                # Standard initialization: optimize full compression tokens
                compression_tokens = self._init_compression_tokens(
                    batch_size,
                    num_compression_tokens,
                    hidden_size,
                    init_method,
                    mvn_dist,
                    pca_components=pca_components,
                    pca_mean=pca_mean,
                    loaded_embeddings=loaded_embeddings,
                )
                # Move to device and save initialization embedding (before optimization)
                # Create new Parameter on device to avoid non-leaf tensor issue
                compression_tokens = torch.nn.Parameter(compression_tokens.data.to(device))
                initialization_embeddings = compression_tokens.detach().clone().cpu()  # [batch, mem, hidden]
                optimizer, lr_scheduler = self._build_optimizer_and_scheduler(
                    [compression_tokens], num_training_steps=self.args.max_optimization_steps_per_sample
                )

            compression_tokens_attention_mask = torch.tensor([[1]], dtype=full_attention_mask.dtype).repeat(
                batch_size, num_compression_tokens
            )

            # Determine maximum effective length present in this batch (exclude padding)
            per_sample_lengths = full_attention_mask.sum(dim=1).tolist()
            max_len = int(max(per_sample_lengths)) if len(per_sample_lengths) > 0 else full_attention_mask.shape[1]
            seq_len = min(min_len, max_len)
            stage_index = 0

            while True:
                # Track if we've reset the scheduler for this stage (to prevent double resets)
                scheduler_reset_used = False
                # Slice to current effective sequence length
                input_ids = full_input_ids[:, :seq_len]
                inputs_embeds = full_model_token_embeddings[:, :seq_len, :]
                target_hidden = list(h[:, :seq_len] for h in target_hidden_full)
                attention_mask = full_attention_mask[:, :seq_len]

                pbar = tqdm(
                    range(self.args.max_optimization_steps_per_token),
                    total=self.args.max_optimization_steps_per_token,
                    # disable=True,
                    leave=False,
                )
                pbar.set_description(f"Stage L={seq_len}")
                last_loss_val = None
                last_conv = None
                steps_taken = 0
                converged = False

                # Training loop - may be repeated once if scheduler reset is enabled
                while True:
                    for i in pbar:
                        # Reconstruct compression tokens from PCA coefficients if using pretrained_pca
                        if init_method == "pretrained_pca":
                            # Reconstruct: [batch, n_components] @ [n_components, flattened_dim] + [flattened_dim] -> [batch, flattened_dim]
                            reconstructed_flat = torch.matmul(
                                pca_coefficients, pca_components_device
                            ) + pca_mean_device.unsqueeze(0)
                            compression_tokens = reconstructed_flat.reshape(batch_size, num_compression_tokens, hidden_size)
                        # else: compression_tokens is already defined in the outer scope

                        current_compression_tokens = compression_tokens.clone()
                        if self.args.low_dim_projection:
                            current_compression_tokens = low_dim_prjoection(compression_tokens)

                        model_tokens_with_compression_tokens = torch.cat(
                            [current_compression_tokens.to(inputs_embeds.device).to(inputs_embeds.dtype), inputs_embeds], dim=1
                        )
                        attention_mask_with_compression_tokens = torch.cat(
                            [compression_tokens_attention_mask, attention_mask], dim=1
                        )
                        # print("input_ids.shape", input_ids.shape)
                        loss, alignment_loss, convergece_per_sample, generated_text, ground_truth_text = self.compute_loss(
                            model,
                            input_ids,
                            inputs_embeds,
                            attention_mask,
                            model_tokens_with_compression_tokens,
                            attention_mask_with_compression_tokens,
                            num_compression_tokens,
                            target_hidden=target_hidden,
                        )
                        loss.backward()
                        steps_taken += 1
                        pbar.update(1)

                        # Get gradient norm from coefficients or compression_tokens
                        if init_method == "pretrained_pca":
                            grad_norm = pca_coefficients.grad.norm(2).item() if pca_coefficients.grad is not None else 0.0
                            comp_mean = compression_tokens.mean().item()
                            comp_std = compression_tokens.std().item()
                        else:
                            grad_norm = compression_tokens.grad.norm(2).item() if compression_tokens.grad is not None else 0.0
                            comp_mean = compression_tokens.mean().item()
                            comp_std = compression_tokens.std().item()

                        log_lr = self.args.learning_rate
                        if lr_scheduler is not None:
                            log_lr = lr_scheduler.get_last_lr()[0]

                        pbar.set_postfix(
                            loss=loss.item(),
                            convergece_per_sample=convergece_per_sample.mean().item(),
                            compr_t_mean=comp_mean,
                            compr_t_std=comp_std,
                            grad=grad_norm,
                            lr=log_lr,
                        )

                        # For logging, use compression_tokens (reconstructed if using PCA)
                        self._log_step(
                            loss,
                            alignment_loss,
                            convergece_per_sample,
                            compression_tokens,
                            lr_scheduler,
                            generated_text,
                            ground_truth_text,
                        )

                        optimizer.step()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                        if self.args.low_dim_projection and self.args.low_dim_proj_train and low_dim_optim is not None:
                            low_dim_optim.step()
                            low_dim_optim.zero_grad()
                            if low_dim_scheduler is not None:
                                low_dim_scheduler.step()

                        last_loss_val = float(loss.item())
                        last_conv = convergece_per_sample.detach().cpu()

                        if convergece_per_sample.mean().item() >= threshold:
                            converged = True
                            break

                    # Check convergence after training loop
                    if converged:
                        break

                    # If not converged and reset is enabled and not yet used, reset scheduler and retry
                    if (
                        not converged
                        and self.args.progressive_reset_lr_scheduler_on_non_convergence
                        and not scheduler_reset_used
                    ):
                        print(f"Not converged at seq_len={seq_len}, resetting LR scheduler and retrying...")
                        # Rebuild scheduler with same parameters
                        if init_method == "pretrained_pca":
                            optimizer, lr_scheduler = self._build_optimizer_and_scheduler([pca_coefficients])
                        else:
                            optimizer, lr_scheduler = self._build_optimizer_and_scheduler(
                                [compression_tokens], num_training_steps=self.args.max_optimization_steps_per_token
                            )
                        scheduler_reset_used = True
                        # Reset progress bar and continue training
                        pbar = tqdm(
                            range(self.args.max_optimization_steps_per_token),
                            total=self.args.max_optimization_steps_per_token,
                            leave=False,
                        )
                        pbar.set_description(f"Stage L={seq_len} (retry)")
                        continue
                    else:
                        # Not converged and either reset disabled or already used - break inner loop
                        break

                # Save snapshot for this stage
                with torch.no_grad():
                    tokenizer = self.processing_class
                    # Reconstruct compression tokens if using PCA (for saving)
                    pca_coefficients_to_save = None
                    if init_method == "pretrained_pca":
                        reconstructed_flat = torch.matmul(pca_coefficients, pca_components_device) + pca_mean_device.unsqueeze(
                            0
                        )
                        pca_coefficients_to_save = pca_coefficients.clone().detach().to(torch.float32).cpu().numpy().tolist()
                        comp_tokens_gpu = reconstructed_flat.reshape(batch_size, num_compression_tokens, hidden_size)
                        comp_tokens_cpu = comp_tokens_gpu.detach().cpu()
                        # For PCA, reconstruct orig from coefficients as well (before optimization would be different, but we use current)
                        orig_comp_tokens_gpu = (
                            comp_tokens_gpu  # Use same for now, or could reconstruct from initial coefficients
                        )
                        orig_comp_tokens_cpu = orig_comp_tokens_gpu.detach().cpu()
                    else:
                        # Reconstruct current compression tokens (after low_dim_projection if applicable)
                        if self.args.low_dim_projection:
                            comp_tokens_gpu = low_dim_prjoection(compression_tokens)
                        else:
                            comp_tokens_gpu = compression_tokens
                        comp_tokens_cpu = comp_tokens_gpu.detach().cpu()
                        orig_comp_tokens_gpu = compression_tokens  # Original before low_dim_projection
                        orig_comp_tokens_cpu = orig_comp_tokens_gpu.detach().cpu()

                    # low_dim_prjoection_w_cpu = None
                    # low_dim_prjoection_b_cpu = None
                    # if self.args.low_dim_projection:
                    #     low_dim_prjoection_w_cpu = low_dim_prjoection.weight.data.cpu()
                    #     low_dim_prjoection_b_cpu = low_dim_prjoection.bias.data.cpu()

                    # Compute per-sample information gain (CE-reduction in bits) with sum reduction
                    # Reconstruct final compression tokens for information gain computation
                    if init_method == "pretrained_pca":
                        final_compression_tokens_for_ig = (
                            torch.matmul(pca_coefficients, pca_components_device) + pca_mean_device.unsqueeze(0)
                        ).reshape(batch_size, num_compression_tokens, hidden_size)
                    else:
                        final_compression_tokens_for_ig = compression_tokens
                    if self.args.low_dim_projection:
                        final_compression_tokens_for_ig = low_dim_prjoection(final_compression_tokens_for_ig)

                    per_sample_info_gain = []
                    for j in range(batch_size):
                        # Extract per-sample data
                        sample_input_ids = input_ids[j : j + 1]  # [1, seq_len]
                        sample_attention_mask = attention_mask[j : j + 1]  # [1, seq_len]
                        sample_compression_tokens = final_compression_tokens_for_ig[j : j + 1]

                        # H_LM for this sample
                        sample_outputs_lm = model(input_ids=sample_input_ids, attention_mask=sample_attention_mask)
                        sample_logits_lm = sample_outputs_lm.logits  # [1, seq_len, vocab_size]

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

                        # H_LM+[mem] for this sample
                        sample_inputs_embeds = inputs_embeds[j : j + 1]
                        sample_model_tokens_with_compression = torch.cat(
                            [
                                sample_compression_tokens.to(sample_inputs_embeds.device).to(sample_inputs_embeds.dtype),
                                sample_inputs_embeds,
                            ],
                            dim=1,
                        )
                        sample_compression_attention_mask = compression_tokens_attention_mask[j : j + 1]
                        sample_attention_mask_with_compression = torch.cat(
                            [sample_compression_attention_mask, sample_attention_mask], dim=1
                        )

                        sample_outputs_mem = model(
                            inputs_embeds=sample_model_tokens_with_compression,
                            attention_mask=sample_attention_mask_with_compression,
                        )
                        sample_logits_mem = sample_outputs_mem.logits  # [1, num_compression_tokens + seq_len, vocab_size]

                        sample_aligned_logits_mem = sample_logits_mem[:, num_compression_tokens:, :]  # [1, seq_len, vocab_size]

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

                        # Per-sample information gain
                        sample_info_gain = sample_H_LM_bits - sample_H_LM_mem_bits
                        per_sample_info_gain.append(sample_info_gain)

                    # Save embeddings to disk in bfloat16 format before converting to fp32
                    embeddings_dir = None
                    if self.args.output_dir:
                        embeddings_dir = os.path.join(self.args.output_dir, "embeddings")
                        os.makedirs(embeddings_dir, exist_ok=True)

                    for j in range(batch_size):
                        attn = attention_mask[j].bool()
                        ids = input_ids[j][attn]
                        text = tokenizer.decode(ids.tolist(), skip_special_tokens=True) if tokenizer is not None else ""
                        sample_id_val = int(sample_id_counter + j)

                        # Save embeddings to disk in bfloat16 before converting to fp32
                        if embeddings_dir is not None and stage_index % 50 == 0:
                            # Get embeddings from GPU tensors and convert to bfloat16 (before moving to CPU and converting to fp32)
                            comp_tokens_bfloat = comp_tokens_gpu[j].to(torch.bfloat16).detach().cpu()
                            orig_comp_tokens_bfloat = orig_comp_tokens_gpu[j].to(torch.bfloat16).detach().cpu()
                            initialization_embedding_bfloat = initialization_embeddings[j].to(torch.bfloat16)

                            # Create unique file names: embedding_sample_{sample_id}_stage_{stage_index}.pt
                            embedding_filename = f"embedding_sample_{sample_id_val}_stage_{stage_index}.pt"
                            orig_embedding_filename = f"orig_embedding_sample_{sample_id_val}_stage_{stage_index}.pt"
                            init_embedding_filename = f"initialization_embedding_sample_{sample_id_val}_stage_{stage_index}.pt"
                            low_dim_proj_filename = f"low_dim_proj_sample_{sample_id_val}_stage_{stage_index}.pt"

                            embedding_path = os.path.join(embeddings_dir, embedding_filename)
                            orig_embedding_path = os.path.join(embeddings_dir, orig_embedding_filename)
                            init_embedding_path = os.path.join(embeddings_dir, init_embedding_filename)
                            low_dim_proj_path = os.path.join(embeddings_dir, low_dim_proj_filename)

                            torch.save(comp_tokens_bfloat, embedding_path)
                            torch.save(orig_comp_tokens_bfloat, orig_embedding_path)
                            torch.save(initialization_embedding_bfloat, init_embedding_path)
                            if self.args.low_dim_projection:
                                torch.save(low_dim_prjoection.state_dict(), low_dim_proj_path)

                        # Convert to fp32 for dataset storage
                        embedding = comp_tokens_cpu[j].to(torch.float32).numpy().tolist()
                        orig_embedding = orig_comp_tokens_cpu[j].to(torch.float32).numpy().tolist()

                        initialization_embedding = initialization_embeddings[j].to(torch.float32).numpy().tolist()
                        # if low_dim_prjoection_w_cpu is not None:
                        #     low_dim_prjoection_w_cpu = low_dim_prjoection_w_cpu.to(torch.float32).numpy().tolist()
                        # if low_dim_prjoection_b_cpu is not None:
                        #     low_dim_prjoection_b_cpu = low_dim_prjoection_b_cpu.to(torch.float32).numpy().tolist()

                        collected_rows.append(
                            {
                                "sample_id": int(sample_id_counter + j),
                                "stage_index": int(stage_index),
                                "stage_seq_len": int(seq_len),
                                "text": text,
                                "embedding": embedding,
                                # "low_dim_prjoection_w": low_dim_prjoection_w_cpu,
                                # "low_dim_prjoection_b": low_dim_prjoection_b_cpu,
                                "orig_embedding": orig_embedding,
                                "pca_coefficients_to_save": pca_coefficients_to_save,
                                "initialization_embedding": initialization_embedding,  # [mem, hidden] - state before optimization
                                "final_loss": (float(last_loss_val) if last_loss_val is not None else None),
                                "final_convergence": (float(last_conv[j].item()) if last_conv is not None else None),
                                "num_input_tokens": int(attn.sum().item()),
                                "num_compression_tokens": int(num_compression_tokens),
                                "hidden_size": int(comp_tokens_cpu.shape[-1]),
                                "loss_type": getattr(self.args, "loss_type", "l2"),
                                "dtype": getattr(self.args, "dtype", "float32"),
                                "model_checkpoint": getattr(self.args, "model_checkpoint", ""),
                                "max_optimization_steps_per_sample": int(
                                    getattr(
                                        self.args,
                                        "max_optimization_steps_per_sample",
                                        0,
                                    )
                                ),
                                "convergence_threshold": float(threshold),
                                "steps_taken": int(steps_taken),
                                "information_gain_bits": float(
                                    per_sample_info_gain[j]
                                ),  # Per-sample info gain in bits (sum reduction)
                            }
                        )

                stage_index += 1
                # Advance to next length or exit
                if seq_len >= max_len:
                    break
                if max_stages_cap and stage_index >= max_stages_cap:
                    break
                if not converged:
                    print("Not converged in max_optimization_steps_per_sample. Stop at seq_len =", seq_len)
                    break
                seq_len = min(seq_len + step_increment, max_len)

            sample_id_counter += batch_size

        # Close writer
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        save_path = self._save_artifacts(None, collected_rows, "progressive_prefixes")
        if save_path is not None:
            return save_path
        return None
