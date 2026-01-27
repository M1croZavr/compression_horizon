from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from compression_horizon.inference.generation import generate_from_compression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate convergence of compression tokens on stored experiments."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="HF checkpoint, e.g. unsloth/Llama-3.2-1B",
    )
    parser.add_argument(
        "--experiment-path",
        required=True,
        help="Path to experiment folder or compressed_prefixes dataset.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Model dtype.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Device selection.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for number of samples.",
    )
    parser.add_argument(
        "--prepend-first-token",
        action="store_true",
        help="Prepend the first token embedding before generation and shift targets.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip greedy generation convergence.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float16":
        return torch.float16
    return torch.float32


def load_experiment_dataset(experiment_path: Path) -> tuple[Path, Path]:
    if (experiment_path / "compressed_prefixes").exists():
        return experiment_path, experiment_path / "compressed_prefixes"
    return experiment_path.parent, experiment_path


def compute_match_rate(
    predicted_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[float, int, int]:
    pred_len = predicted_ids.shape[1]
    target_len = target_ids.shape[1]
    min_len = min(pred_len, target_len)
    if min_len == 0:
        return float("nan"), pred_len, target_len
    match_rate = (predicted_ids[:, :min_len] == target_ids[:, :min_len]).float().mean().item()
    return match_rate, pred_len, target_len


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype)
    experiment_path = Path(args.experiment_path)
    base_path, dataset_path = load_experiment_dataset(experiment_path)

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    experiment_result = load_from_disk(str(dataset_path))

    inference_convergences: list[float] = []
    generation_convergences: list[float] = []

    sample_ids = experiment_result["sample_id"]
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]

    model.eval()
    for sample_id in sample_ids:
        compression_embeddings = torch.load(
            base_path / f"compression_token_embeddings_{sample_id}.pt",
            map_location=device,
        ).unsqueeze(dim=0)

        sample = experiment_result[sample_id]
        max_length = sample["num_input_tokens"]
        tokenized_sample = tokenizer(
            sample["text"],
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = tokenized_sample["input_ids"].to(device)
        attention_mask = tokenized_sample["attention_mask"].to(device)
        with torch.no_grad():
            sequence_embeddings = model.get_input_embeddings()(input_ids)

        with torch.no_grad():
            outputs = model(
                inputs_embeds=torch.cat((compression_embeddings, sequence_embeddings), dim=1),
                attention_mask=torch.cat(
                    (
                        torch.ones(
                            1,
                            compression_embeddings.size(1),
                            dtype=torch.long,
                            device=device,
                        ),
                        attention_mask,
                    ),
                    dim=1,
                ),
            )
        logits = outputs.logits[:, :-1]
        if logits.shape[1] != input_ids.shape[1]:
            logits = logits[:, -input_ids.shape[1] :]
        predicted_ids = logits.argmax(-1)
        inference_convergence, _, _ = compute_match_rate(predicted_ids, input_ids)
        inference_convergences.append(inference_convergence)

        train_convergence = sample.get("final_convergence")
        train_error = None if train_convergence is None else 1.0 - train_convergence
        inference_error = 1.0 - inference_convergence
        gap = None if train_convergence is None else inference_convergence - train_convergence

        print(f"sample_id={sample_id}")
        print(
            "train_convergence="
            f"{train_convergence} train_error={train_error} "
            f"inference_convergence={inference_convergence:.6f} "
            f"inference_error={inference_error:.6f} gap={gap}"
        )

        if args.skip_generation:
            continue

        if args.prepend_first_token:
            prefix_embeddings = torch.cat(
                (compression_embeddings, model.get_input_embeddings()(input_ids[:, :1])),
                dim=1,
            )
            target_ids = input_ids[:, 1:]
            max_new_tokens = max_length - 1
        else:
            prefix_embeddings = compression_embeddings
            target_ids = input_ids
            max_new_tokens = max_length

        try:
            _, generated_ids = generate_from_compression(
                model,
                tokenizer,
                prefix_embeddings,
                max_new_tokens,
                1,
                return_generated_ids=True,
            )
            generation_convergence, pred_len, target_len = compute_match_rate(
                generated_ids,
                target_ids,
            )
            generation_convergences.append(generation_convergence)
            generation_error = 1.0 - generation_convergence
            gap = None if train_convergence is None else generation_convergence - train_convergence
            print(
                "generation_convergence="
                f"{generation_convergence:.6f} generation_error={generation_error:.6f} "
                f"gap={gap} length={pred_len}/{target_len}"
            )
        except RuntimeError as exc:
            print(f"generation_error={exc}")

    if inference_convergences:
        print(f"inference_convergence_mean={mean(inference_convergences):.6f}")
        print(f"inference_error_mean={mean([1.0 - c for c in inference_convergences]):.6f}")
    if generation_convergences:
        print(f"generation_convergence_mean={mean(generation_convergences):.6f}")
        print(f"generation_error_mean={mean([1.0 - c for c in generation_convergences]):.6f}")


if __name__ == "__main__":
    main()
