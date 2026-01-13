import argparse
import os
from typing import Optional

from datasets import Dataset, load_dataset
from openai import OpenAI
from transformers import AutoTokenizer


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
) -> str:
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
        Generated paraphrase
    """
    if paraphrase_type == "full":
        system_prompt = (
            "You are a helpful assistant that paraphrases text while preserving the original meaning, "
            "style, and tone. Generate a high-quality paraphrase of the given text."
        )
        user_prompt = f"Paraphrase the following text while maintaining its meaning and style:\n\n{text}"
    elif paraphrase_type == "partial":
        # Extract prefix (first prefix_tokens tokens) and remainder
        tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
        prefix_tokens_list = tokens[:prefix_tokens]
        remainder_tokens_list = tokens[prefix_tokens:]
        prefix_text = tokenizer.decode(prefix_tokens_list, skip_special_tokens=True)
        remainder_text = tokenizer.decode(remainder_tokens_list, skip_special_tokens=True)

        system_prompt = (
            "You are a helpful assistant that paraphrases text while preserving the original meaning, "
            "style, and tone. You must start your response with the exact prefix text provided, "
            "then continue with a paraphrased version of the remainder text."
        )
        user_prompt = (
            f"Generate a paraphrase of the following text with these requirements:\n\n"
            f"1. Start your response with this EXACT prefix (copy it word-for-word):\n{prefix_text}\n\n"
            f"2. Then continue by paraphrasing this remainder text:\n{remainder_text}\n\n"
            f"3. The paraphrase should maintain the original meaning and style.\n\n"
            f"Full original text for context:\n{text}"
        )
    else:
        raise ValueError(f"Unknown paraphrase_type: {paraphrase_type}")

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=2500,
                temperature=0.5,
                presence_penalty=0,
                top_p=0.95,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating paraphrase: {e}")
            return ""


def generate_paraphrases_for_dataset(
    dataset: Dataset,
    client: OpenAI,
    model: str,
    tokenizer: AutoTokenizer,
    output_dir: str,
    prefix_tokens: int = 64,
):
    """Generate full and partial paraphrases for all samples in the dataset."""
    os.makedirs(output_dir, exist_ok=True)

    full_paraphrase_results = []
    partial_paraphrase_results = []
    total_samples = len(dataset)

    for idx, example in enumerate(dataset):
        print(f"\nProcessing sample {idx + 1}/{total_samples}")
        truncated_text = example["truncated_text"]

        # Generate full paraphrase
        print("Generating full paraphrase...")
        full_paraphrase = generate_paraphrase(
            client=client,
            model=model,
            text=truncated_text,
            paraphrase_type="full",
        )

        # Generate partial paraphrase
        print("Generating partial paraphrase...")
        partial_paraphrase = generate_paraphrase(
            client=client,
            model=model,
            text=truncated_text,
            paraphrase_type="partial",
            tokenizer=tokenizer,
            prefix_tokens=prefix_tokens,
        )

        # Add to full paraphrase dataset
        full_result = {
            "sample_id": idx,
            "text": full_paraphrase,
            "original_text": example["original_text"],
            "truncated_text": truncated_text,
            "num_tokens": example["num_tokens"],
        }
        full_paraphrase_results.append(full_result)

        # Add to partial paraphrase dataset
        partial_result = {
            "sample_id": idx,
            "text": partial_paraphrase,
            "original_text": example["original_text"],
            "truncated_text": truncated_text,
            "num_tokens": example["num_tokens"],
        }
        partial_paraphrase_results.append(partial_result)

        # Save intermediate results periodically
        if (idx + 1) % 10 == 0:
            full_intermediate = Dataset.from_list(full_paraphrase_results)
            partial_intermediate = Dataset.from_list(partial_paraphrase_results)
            full_intermediate_path = os.path.join(output_dir, "full_paraphrases_intermediate")
            partial_intermediate_path = os.path.join(output_dir, "partial_paraphrases_intermediate")
            full_intermediate.save_to_disk(full_intermediate_path)
            partial_intermediate.save_to_disk(partial_intermediate_path)
            print(f"Saved intermediate results ({idx + 1} samples)")

    # Save final results - separate datasets
    full_dataset = Dataset.from_list(full_paraphrase_results)
    partial_dataset = Dataset.from_list(partial_paraphrase_results)

    full_path = os.path.join(output_dir, "full_paraphrases")
    partial_path = os.path.join(output_dir, "partial_paraphrases")

    full_dataset.save_to_disk(full_path)
    partial_dataset.save_to_disk(partial_path)

    print(f"\nSaved full paraphrases to {full_path} ({len(full_paraphrase_results)} samples)")
    print(f"Saved partial paraphrases to {partial_path} ({len(partial_paraphrase_results)} samples)")

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

    args = parser.parse_args()

    # Check for API key
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
    print(f"Using model: {args.model}")
    print(f"Output directory: {args.output_dir}")

    # Generate paraphrases
    full_dataset, partial_dataset = generate_paraphrases_for_dataset(
        dataset=dataset,
        client=client,
        model=args.model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        prefix_tokens=args.prefix_tokens,
    )

    print("\nSummary:")
    print(f"  Full paraphrases: {len(full_dataset)} samples")
    print(f"  Partial paraphrases: {len(partial_dataset)} samples")
    print("\nDone!")


if __name__ == "__main__":
    main()
