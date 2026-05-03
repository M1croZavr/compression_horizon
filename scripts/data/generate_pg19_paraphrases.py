import argparse
import os
import random
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Example run:
# python scripts/data/generate_pg19_paraphrases.py \
#     --tokenizer unsloth/Meta-Llama-3.1-8B \
#     --model openai/gpt-oss-120b \
#     --max_tokens 4096 \
#     --prefix_tokens 64 \
#     --limit 10 \
#     --push_to_hub \
#     --datasets lowercased_partial \
#     --output_dir artifacts/pg19_paraphrases \
#     --hub_dataset_id_lowercased_partial mrsndmn/pg19-partial-lowercased-4096-tokens

# python scripts/data/generate_pg19_paraphrases.py \
#     --tokenizer unsloth/Meta-Llama-3.1-8B \
#     --model openai/gpt-oss-120b \
#     --max_tokens 4096 \
#     --prefix_tokens 64 \
#     --limit 10 \
#     --push_to_hub \
#     --datasets random_suffix_shuffle \
#     --output_dir artifacts/pg19_paraphrases \
#     --hub_dataset_id_random_suffix_shuffle mrsndmn/pg19-random-suffix-shuffle-4096


def load_and_tokenize_dataset(
    dataset_name: str,
    tokenizer_name: str,
    max_tokens: int = 256,
    split: str = "test",
    limit: Optional[int] = None,
):
    """Load dataset and tokenize text, truncating to max_tokens."""
    print(f"Loading dataset: {dataset_name}")
    raw_dataset = load_dataset(dataset_name, split=split)

    if limit is not None:
        raw_dataset = raw_dataset.select(range(limit))

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_and_truncate(example):
        """Tokenize text and truncate to max_tokens."""
        text = example.get("text", "")
        # Tokenize
        tokens = tokenizer(text, truncation=True, max_length=max_tokens, return_tensors="pt")
        input_ids = tokens["input_ids"][0]
        # Decode back to text (this gives us the truncated text)
        truncated_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        return {
            "original_text": text,
            "truncated_text": truncated_text,
            "num_tokens": len(input_ids),
        }

    print("Tokenizing and truncating dataset...")
    dataset = raw_dataset.map(tokenize_and_truncate, remove_columns=raw_dataset.column_names)
    return dataset, tokenizer


def generate_paraphrase(
    client: OpenAI,
    model: str,
    text: str,
    paraphrase_type: str = "full",
    tokenizer: Optional[AutoTokenizer] = None,
    prefix_tokens: int = 64,
) -> tuple[str, Optional[str]]:
    """
    Generate a paraphrase using the LLM API.

    Args:
        client: OpenAI client instance
        model: Model name to use
        text: Text to paraphrase
        paraphrase_type: "full" or "partial"
        tokenizer: Tokenizer for partial paraphrases (to extract prefix)
        prefix_tokens: Number of tokens to keep as prefix for partial paraphrases

    Returns:
        For "full": (paraphrase, None)
        For "partial": (paraphrased_remainder, prefix_text) - prefix should be prepended programmatically
    """
    if paraphrase_type == "full":
        system_prompt = (
            "You are a helpful assistant that paraphrases text while preserving the original meaning, "
            "style, and tone. Generate a high-quality paraphrase of the given text."
        )
        user_prompt = f"Paraphrase the following text while maintaining its meaning and style. DO NOT MODIFY text syle. No markdown headers and fotmatting. Text: \n\n{text}"
        prefix_text = None
    elif paraphrase_type == "partial":
        # Extract prefix (first prefix_tokens tokens) and remainder
        tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
        prefix_tokens_list = tokens[:prefix_tokens]
        remainder_tokens_list = tokens[prefix_tokens:]
        prefix_text = tokenizer.decode(prefix_tokens_list, skip_special_tokens=True)
        remainder_text = tokenizer.decode(remainder_tokens_list, skip_special_tokens=True)

        system_prompt = (
            "You are a helpful assistant that paraphrases text while preserving the original meaning, "
            "style, and tone. Paraphrase only the given text, do not include any prefix."
        )
        user_prompt = f"Context:\n{prefix_text}\n\nParaphrase the following text while maintaining its meaning and style:\n\n{remainder_text}"
    else:
        raise ValueError(f"Unknown paraphrase_type: {paraphrase_type}")

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=2500,
            temperature=0.0,
            presence_penalty=0,
            top_p=0.95,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if content is None:
            print(f"Warning: API returned None content for {paraphrase_type} paraphrase")
            return ("", prefix_text)
        result = content.strip()
        if not result:
            print(f"Warning: API returned empty content for {paraphrase_type} paraphrase")
        else:
            print(f"Successfully generated {paraphrase_type} paraphrase ({len(result)} chars)")
        return (result, prefix_text)
    except Exception as e:
        print(f"Error generating {paraphrase_type} paraphrase: {e}")
        import traceback

        traceback.print_exc()
        return ("", prefix_text)


def create_lowercased_dataset(
    dataset: Dataset,
    output_dir: str,
    push_to_hub: bool = False,
    hub_dataset_id_lowercased: Optional[str] = None,
):
    """Create a lowercased version of the original dataset."""
    os.makedirs(output_dir, exist_ok=True)

    def lowercase_text(example):
        """Lowercase the text column."""
        return {
            "text": example["original_text"].lower(),
            "original_text": example["original_text"],
            "truncated_text": example["truncated_text"],
            "num_tokens": example["num_tokens"],
        }

    print("Creating lowercased dataset...")
    lowercased_dataset = dataset.map(lowercase_text)

    lowercased_path = os.path.join(output_dir, "lowercased_original")
    lowercased_dataset.save_to_disk(lowercased_path)
    print(f"Saved lowercased dataset to {lowercased_path} ({len(lowercased_dataset)} samples)")

    # Push to hub if requested
    if push_to_hub and hub_dataset_id_lowercased:
        print(f"\nPushing lowercased dataset to hub: {hub_dataset_id_lowercased}")
        lowercased_dataset.push_to_hub(hub_dataset_id_lowercased)
        print(f"Successfully pushed lowercased dataset to {hub_dataset_id_lowercased}")
    elif push_to_hub and not hub_dataset_id_lowercased:
        print("Warning: push_to_hub is True but hub_dataset_id_lowercased is not specified, skipping...")

    return lowercased_dataset


def create_lowercased_partial_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    prefix_tokens: int,
    output_dir: str,
    push_to_hub: bool = False,
    hub_dataset_id_lowercased_partial: Optional[str] = None,
):
    """Create a lowercased partial version: keep first prefix_tokens unchanged, lowercase remainder."""
    os.makedirs(output_dir, exist_ok=True)

    def lowercase_partial_text(example):
        """Keep first prefix_tokens unchanged, lowercase the remainder."""
        truncated_text = example["truncated_text"]
        # Tokenize to split into prefix and remainder
        tokens = tokenizer(truncated_text, return_tensors="pt")["input_ids"][0]
        prefix_tokens_list = tokens[:prefix_tokens]
        remainder_tokens_list = tokens[prefix_tokens:]
        # Decode prefix and remainder separately
        prefix_text = tokenizer.decode(prefix_tokens_list, skip_special_tokens=True)
        remainder_text = tokenizer.decode(remainder_tokens_list, skip_special_tokens=True)
        # Lowercase only the remainder
        lowercased_remainder = remainder_text.lower()
        # Concatenate prefix + lowercased remainder
        lowercased_partial_text = prefix_text + lowercased_remainder
        return {
            "text": lowercased_partial_text,
            "original_text": example["original_text"],
            "truncated_text": truncated_text,
            "num_tokens": example["num_tokens"],
        }

    print("Creating lowercased partial dataset...")
    lowercased_partial_dataset = dataset.map(lowercase_partial_text)

    lowercased_partial_path = os.path.join(output_dir, "lowercased_partial")
    lowercased_partial_dataset.save_to_disk(lowercased_partial_path)
    print(f"Saved lowercased partial dataset to {lowercased_partial_path} ({len(lowercased_partial_dataset)} samples)")

    # Push to hub if requested
    if push_to_hub and hub_dataset_id_lowercased_partial:
        print(f"\nPushing lowercased partial dataset to hub: {hub_dataset_id_lowercased_partial}")
        lowercased_partial_dataset.push_to_hub(hub_dataset_id_lowercased_partial)
        print(f"Successfully pushed lowercased partial dataset to {hub_dataset_id_lowercased_partial}")
    elif push_to_hub and not hub_dataset_id_lowercased_partial:
        print("Warning: push_to_hub is True but hub_dataset_id_lowercased_partial is not specified, skipping...")

    return lowercased_partial_dataset


def create_random_suffix_shuffle_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    prefix_tokens: int,
    output_dir: str,
    push_to_hub: bool = False,
    hub_dataset_id_random_suffix_shuffle: Optional[str] = None,
):
    """Create a random suffix shuffle version: keep first prefix_tokens unchanged, shuffle words in remainder."""
    os.makedirs(output_dir, exist_ok=True)

    def shuffle_suffix_words(example):
        """Keep first prefix_tokens unchanged, shuffle words in the remainder."""
        truncated_text = example["truncated_text"]
        # Tokenize to split into prefix and remainder
        tokens = tokenizer(truncated_text, return_tensors="pt")["input_ids"][0]
        prefix_tokens_list = tokens[:prefix_tokens]
        remainder_tokens_list = tokens[prefix_tokens:]
        # Decode prefix and remainder separately
        prefix_text = tokenizer.decode(prefix_tokens_list, skip_special_tokens=True)
        remainder_text = tokenizer.decode(remainder_tokens_list, skip_special_tokens=True)
        # Split remainder into words and shuffle
        remainder_words = remainder_text.split()
        random.shuffle(remainder_words)
        # Join shuffled words back together
        shuffled_remainder = " ".join(remainder_words)
        # Concatenate prefix + shuffled remainder (add space if prefix doesn't end with one)
        if prefix_text and not prefix_text.endswith(" "):
            shuffled_text = prefix_text + " " + shuffled_remainder
        else:
            shuffled_text = prefix_text + shuffled_remainder
        return {
            "text": shuffled_text,
            "original_text": example["original_text"],
            "truncated_text": truncated_text,
            "num_tokens": example["num_tokens"],
        }

    print("Creating random suffix shuffle dataset...")
    shuffled_dataset = dataset.map(shuffle_suffix_words)

    shuffled_path = os.path.join(output_dir, "random_suffix_shuffle")
    shuffled_dataset.save_to_disk(shuffled_path)
    print(f"Saved random suffix shuffle dataset to {shuffled_path} ({len(shuffled_dataset)} samples)")

    # Push to hub if requested
    if push_to_hub and hub_dataset_id_random_suffix_shuffle:
        print(f"\nPushing random suffix shuffle dataset to hub: {hub_dataset_id_random_suffix_shuffle}")
        shuffled_dataset.push_to_hub(hub_dataset_id_random_suffix_shuffle)
        print(f"Successfully pushed random suffix shuffle dataset to {hub_dataset_id_random_suffix_shuffle}")
    elif push_to_hub and not hub_dataset_id_random_suffix_shuffle:
        print("Warning: push_to_hub is True but hub_dataset_id_random_suffix_shuffle is not specified, skipping...")

    return shuffled_dataset


def create_model_sampled_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prefix_tokens: int,
    max_tokens: int,
    output_dir: str,
    push_to_hub: bool = False,
    hub_dataset_id_model_sampled: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 16,
    temperature: Optional[float] = None,
):
    """Create a model_sampled version: use first prefix_tokens as prefix, generate continuation with greedy or temperature sampling."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model.to(device)

    print("Creating model_sampled dataset...")
    results = []
    total_samples = len(dataset)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    total_batches = (total_samples + batch_size - 1) // batch_size

    # Process in batches
    for batch_start in tqdm(range(0, total_samples, batch_size), desc="Generating batches", total=total_batches):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_examples = dataset[batch_start:batch_end]
        current_batch_size = batch_end - batch_start

        # Extract prefix token IDs for all examples in batch
        # Note: dataset[batch_start:batch_end] returns a dict with column names as keys and lists as values
        prefix_input_ids_list = []
        batch_metadata = []

        for idx in range(current_batch_size):
            truncated_text = batch_examples["truncated_text"][idx]
            # Tokenize to extract prefix token IDs directly
            tokens = tokenizer(truncated_text, return_tensors="pt")["input_ids"][0]
            prefix_tokens_list = tokens[:prefix_tokens]
            prefix_input_ids_list.append(prefix_tokens_list)
            batch_metadata.append(
                {
                    "original_text": batch_examples["original_text"][idx],
                    "truncated_text": truncated_text,
                    "num_tokens": batch_examples["num_tokens"][idx],
                }
            )

        # Pad sequences to the same length for batching
        max_prefix_len = max(len(seq) for seq in prefix_input_ids_list)
        padded_prefix_ids = []
        attention_masks = []

        for prefix_ids in prefix_input_ids_list:
            padding_length = max_prefix_len - len(prefix_ids)
            padded = torch.cat([prefix_ids, torch.full((padding_length,), pad_token_id, dtype=prefix_ids.dtype)])
            padded_prefix_ids.append(padded)
            attention_mask = torch.cat(
                [torch.ones(len(prefix_ids), dtype=torch.bool), torch.zeros(padding_length, dtype=torch.bool)]
            )
            attention_masks.append(attention_mask)

        # Stack into batch tensor
        batch_input_ids = torch.stack(padded_prefix_ids).to(device)
        batch_attention_mask = torch.stack(attention_masks).to(device)

        # Generate continuation with greedy or temperature sampling for entire batch
        generate_kwargs = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "max_new_tokens": max_tokens,
            "pad_token_id": pad_token_id,
        }
        if temperature is not None and temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["do_sample"] = False  # Greedy sampling

        with torch.no_grad():
            generated_output = model.generate(**generate_kwargs)

        # Decode each generated sequence
        for idx in range(current_batch_size):
            generated_text = tokenizer.decode(generated_output[idx], skip_special_tokens=True)
            results.append(
                {
                    "text": generated_text,
                    "original_text": batch_metadata[idx]["original_text"],
                    "truncated_text": batch_metadata[idx]["truncated_text"],
                    "num_tokens": batch_metadata[idx]["num_tokens"],
                }
            )

    # Create dataset from results
    model_sampled_dataset = Dataset.from_list(results, split="test")

    model_sampled_path = os.path.join(output_dir, "model_sampled")
    model_sampled_dataset.save_to_disk(model_sampled_path)
    print(f"Saved model_sampled dataset to {model_sampled_path} ({len(model_sampled_dataset)} samples)")

    # Push to hub if requested
    if push_to_hub and hub_dataset_id_model_sampled:
        print(f"\nPushing model_sampled dataset to hub: {hub_dataset_id_model_sampled}")
        model_sampled_dataset.push_to_hub(hub_dataset_id_model_sampled)
        print(f"Successfully pushed model_sampled dataset to {hub_dataset_id_model_sampled}")
    elif push_to_hub and not hub_dataset_id_model_sampled:
        print("Warning: push_to_hub is True but hub_dataset_id_model_sampled is not specified, skipping...")

    return model_sampled_dataset


def generate_paraphrases_for_dataset(
    dataset: Dataset,
    client: OpenAI,
    model: str,
    tokenizer: AutoTokenizer,
    output_dir: str,
    prefix_tokens: int = 64,
    push_to_hub: bool = False,
    hub_dataset_id_full: Optional[str] = None,
    hub_dataset_id_partial: Optional[str] = None,
    generate_full: bool = True,
    generate_partial: bool = True,
):
    """Generate full and/or partial paraphrases for all samples in the dataset."""
    os.makedirs(output_dir, exist_ok=True)

    full_paraphrase_results = []
    partial_paraphrase_results = []
    total_samples = len(dataset)

    for idx, example in enumerate(dataset):
        print(f"\nProcessing sample {idx + 1}/{total_samples}")
        truncated_text = example["truncated_text"]

        # Generate full paraphrase
        full_paraphrase = None
        if generate_full:
            print("Generating full paraphrase...")
            full_paraphrase, _ = generate_paraphrase(
                client=client,
                model=model,
                text=truncated_text,
                paraphrase_type="full",
            )

            if not full_paraphrase:
                print(f"Warning: Empty full paraphrase for sample {idx}")
                full_paraphrase = None  # Store None to indicate failure

        # Generate partial paraphrase
        partial_remainder = None
        prefix_text = None
        if generate_partial:
            print("Generating partial paraphrase...")
            partial_remainder, prefix_text = generate_paraphrase(
                client=client,
                model=model,
                text=truncated_text,
                paraphrase_type="partial",
                tokenizer=tokenizer,
                prefix_tokens=prefix_tokens,
            )

        # Programmatically prepend prefix to the paraphrased remainder
        partial_paraphrase = None
        if generate_partial:
            if partial_remainder and prefix_text:
                partial_paraphrase = prefix_text + partial_remainder
            elif not partial_remainder:
                print(f"Warning: Empty partial paraphrase remainder for sample {idx}")
                partial_paraphrase = None  # Store None to indicate failure
            else:
                partial_paraphrase = None

        # Add to full paraphrase dataset
        if generate_full:
            # Use empty string instead of None to avoid null values in dataset
            full_result = {
                "sample_id": idx,
                "text": full_paraphrase if full_paraphrase else "",
                "original_text": example["original_text"],
                "truncated_text": truncated_text,
                "num_tokens": example["num_tokens"],
            }
            # Debug: verify text is not None
            if full_result["text"] is None:
                print(f"ERROR: full_result['text'] is None for sample {idx}, converting to empty string")
                full_result["text"] = ""
            full_paraphrase_results.append(full_result)

        # Add to partial paraphrase dataset
        if generate_partial:
            # Use empty string instead of None to avoid null values in dataset
            partial_result = {
                "sample_id": idx,
                "text": partial_paraphrase if partial_paraphrase else "",
                "original_text": example["original_text"],
                "truncated_text": truncated_text,
                "num_tokens": example["num_tokens"],
            }
            # Debug: verify text is not None
            if partial_result["text"] is None:
                print(f"ERROR: partial_result['text'] is None for sample {idx}, converting to empty string")
                partial_result["text"] = ""
            partial_paraphrase_results.append(partial_result)

        # Save intermediate results periodically
        if (idx + 1) % 10 == 0:
            if generate_full and full_paraphrase_results:
                full_intermediate = Dataset.from_list(full_paraphrase_results)
                full_intermediate_path = os.path.join(output_dir, "full_paraphrases_intermediate")
                full_intermediate.save_to_disk(full_intermediate_path)
            if generate_partial and partial_paraphrase_results:
                partial_intermediate = Dataset.from_list(partial_paraphrase_results)
                partial_intermediate_path = os.path.join(output_dir, "partial_paraphrases_intermediate")
                partial_intermediate.save_to_disk(partial_intermediate_path)
            print(f"Saved intermediate results ({idx + 1} samples)")

    # Save final results - separate datasets
    full_dataset = None
    partial_dataset = None

    if generate_full and full_paraphrase_results:
        full_dataset = Dataset.from_list(full_paraphrase_results)
        full_path = os.path.join(output_dir, "full_paraphrases")
        full_dataset.save_to_disk(full_path)
        print(f"\nSaved full paraphrases to {full_path} ({len(full_paraphrase_results)} samples)")

    if generate_partial and partial_paraphrase_results:
        partial_dataset = Dataset.from_list(partial_paraphrase_results)
        partial_path = os.path.join(output_dir, "partial_paraphrases")
        partial_dataset.save_to_disk(partial_path)
        print(f"\nSaved partial paraphrases to {partial_path} ({len(partial_paraphrase_results)} samples)")

    # Push to hub if requested
    if push_to_hub:
        if generate_full and hub_dataset_id_full and full_dataset:
            print(f"\nPushing full paraphrases to hub: {hub_dataset_id_full}")
            full_dataset.push_to_hub(hub_dataset_id_full)
            print(f"Successfully pushed full paraphrases to {hub_dataset_id_full}")
        elif generate_full and push_to_hub and not hub_dataset_id_full:
            print("Warning: push_to_hub is True but hub_dataset_id_full is not specified, skipping...")

        if generate_partial and hub_dataset_id_partial and partial_dataset:
            print(f"\nPushing partial paraphrases to hub: {hub_dataset_id_partial}")
            partial_dataset.push_to_hub(hub_dataset_id_partial)
            print(f"Successfully pushed partial paraphrases to {hub_dataset_id_partial}")
        elif generate_partial and push_to_hub and not hub_dataset_id_partial:
            print("Warning: push_to_hub is True but hub_dataset_id_partial is not specified, skipping...")

    return full_dataset, partial_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate paraphrases for pg19 dataset")
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Tokenizer name or path (e.g., 'HuggingFaceTB/SmolLM2-135M')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model name for paraphrasing (default: 'openai/gpt-oss-120b')",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to truncate text to (default: 256)",
    )
    parser.add_argument(
        "--prefix_tokens",
        type=int,
        default=64,
        help="Number of tokens to keep as prefix for partial paraphrases (default: 64)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (default: None, process all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/pg19_paraphrases",
        help="Output directory for paraphrases (default: 'artifacts/pg19_paraphrases')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: 'test')",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Push generated datasets to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_dataset_id_full",
        type=str,
        default=None,
        help="HuggingFace Hub dataset ID for full paraphrases (e.g., 'username/dataset-full-paraphrases')",
    )
    parser.add_argument(
        "--hub_dataset_id_partial",
        type=str,
        default=None,
        help="HuggingFace Hub dataset ID for partial paraphrases (e.g., 'username/dataset-partial-paraphrases')",
    )
    parser.add_argument(
        "--hub_dataset_id_lowercased",
        type=str,
        default=None,
        help="HuggingFace Hub dataset ID for lowercased original dataset (e.g., 'username/dataset-lowercased')",
    )
    parser.add_argument(
        "--hub_dataset_id_lowercased_partial",
        type=str,
        default=None,
        help="HuggingFace Hub dataset ID for lowercased partial dataset (e.g., 'username/dataset-lowercased-partial')",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="lowercased,lowercased_partial,full_paraphrases,partial_paraphrases,random_suffix_shuffle",
        help="Comma-separated list of dataset types to generate. Options: lowercased, lowercased_partial, full_paraphrases, partial_paraphrases, random_suffix_shuffle, model_sampled (default: all)",
    )
    parser.add_argument(
        "--hub_dataset_id_random_suffix_shuffle",
        type=str,
        default=None,
        help="HuggingFace Hub dataset ID for random suffix shuffle dataset (e.g., 'username/dataset-random-suffix-shuffle')",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Model checkpoint for model_sampled dataset generation (e.g., 'HuggingFaceTB/SmolLM2-135M')",
    )
    parser.add_argument(
        "--hub_dataset_id_model_sampled",
        type=str,
        default=None,
        help="HuggingFace Hub dataset ID for model_sampled dataset (e.g., 'username/dataset-model-sampled')",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for sampling in model_sampled dataset generation. If None or 0, uses greedy sampling (default: None)",
    )

    args = parser.parse_args()

    # Parse and validate dataset types
    valid_datasets = {
        "lowercased",
        "lowercased_partial",
        "full_paraphrases",
        "partial_paraphrases",
        "random_suffix_shuffle",
        "model_sampled",
    }
    requested_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    invalid_datasets = set(requested_datasets) - valid_datasets
    if invalid_datasets:
        raise ValueError(f"Invalid dataset types: {invalid_datasets}. Valid options: {valid_datasets}")
    datasets_to_generate = set(requested_datasets)
    print(f"Datasets to generate: {datasets_to_generate}")

    # Check for API key (only needed for paraphrases)
    client = None
    if "full_paraphrases" in datasets_to_generate or "partial_paraphrases" in datasets_to_generate:
        api_key = os.environ.get("API_KEY")
        if not api_key:
            raise ValueError("API_KEY environment variable is not set")
        # Initialize OpenAI client
        url = "https://foundation-models.api.cloud.ru/v1"
        client = OpenAI(api_key=api_key, base_url=url)

    # Load and tokenize dataset
    dataset, tokenizer = load_and_tokenize_dataset(
        dataset_name="mrsndmn/pg19",
        tokenizer_name=args.tokenizer,
        max_tokens=args.max_tokens,
        split=args.split,
        limit=args.limit,
    )

    print(f"\nDataset loaded: {len(dataset)} samples")
    if "full_paraphrases" in datasets_to_generate or "partial_paraphrases" in datasets_to_generate:
        print(f"Using model: {args.model}")
    print(f"Output directory: {args.output_dir}")

    full_dataset = None
    partial_dataset = None

    # Create lowercased dataset
    if "lowercased" in datasets_to_generate:
        create_lowercased_dataset(
            dataset=dataset,
            output_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            hub_dataset_id_lowercased=args.hub_dataset_id_lowercased,
        )

    # Create lowercased partial dataset
    if "lowercased_partial" in datasets_to_generate:
        create_lowercased_partial_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            prefix_tokens=args.prefix_tokens,
            output_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            hub_dataset_id_lowercased_partial=args.hub_dataset_id_lowercased_partial,
        )

    # Create random suffix shuffle dataset
    if "random_suffix_shuffle" in datasets_to_generate:
        create_random_suffix_shuffle_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            prefix_tokens=args.prefix_tokens,
            output_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            hub_dataset_id_random_suffix_shuffle=args.hub_dataset_id_random_suffix_shuffle,
        )

    # Create model_sampled dataset
    if "model_sampled" in datasets_to_generate:
        if not args.model_checkpoint:
            raise ValueError("--model_checkpoint is required when generating model_sampled dataset")
        print(f"Loading model for generation: {args.model_checkpoint}")
        generation_model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint)
        create_model_sampled_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            model=generation_model,
            prefix_tokens=args.prefix_tokens,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            hub_dataset_id_model_sampled=args.hub_dataset_id_model_sampled,
            batch_size=8,
            temperature=args.temperature,
        )

    # Generate paraphrases
    if "full_paraphrases" in datasets_to_generate or "partial_paraphrases" in datasets_to_generate:
        full_dataset, partial_dataset = generate_paraphrases_for_dataset(
            dataset=dataset,
            client=client,
            model=args.model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            prefix_tokens=args.prefix_tokens,
            push_to_hub=args.push_to_hub,
            hub_dataset_id_full=args.hub_dataset_id_full if "full_paraphrases" in datasets_to_generate else None,
            hub_dataset_id_partial=args.hub_dataset_id_partial if "partial_paraphrases" in datasets_to_generate else None,
            generate_full="full_paraphrases" in datasets_to_generate,
            generate_partial="partial_paraphrases" in datasets_to_generate,
        )

    print("\nSummary:")
    if "lowercased" in datasets_to_generate:
        print("  Lowercased dataset: generated")
    if "lowercased_partial" in datasets_to_generate:
        print("  Lowercased partial dataset: generated")
    if "random_suffix_shuffle" in datasets_to_generate:
        print("  Random suffix shuffle dataset: generated")
    if "model_sampled" in datasets_to_generate:
        print("  Model sampled dataset: generated")
    if "full_paraphrases" in datasets_to_generate and full_dataset:
        print(f"  Full paraphrases: {len(full_dataset)} samples")
    if "partial_paraphrases" in datasets_to_generate and partial_dataset:
        print(f"  Partial paraphrases: {len(partial_dataset)} samples")
    print("\nDone!")


if __name__ == "__main__":
    main()
