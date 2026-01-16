"""Generate text using greedy decoding from progressive training artifacts."""

import argparse
import json
import os
import sys
from typing import Optional

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from compression_horizon.inference.generation import generate_from_compression

# ANSI color codes
RED = "\033[91m"
RESET = "\033[0m"


@torch.no_grad()
def generate_with_token_tracking(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    compressed_embeddings: torch.Tensor,  # [1, mem, hidden]
    max_new_tokens: int,
    stage_seq_len: int,
    only_new_tokens: bool = False,
    show_progress: bool = True,
    stream_tokens: bool = False,
) -> tuple[str, list[int], list[int]]:
    """
    Generate text and return both the text and a list indicating which token positions
    are out of bounds (beyond stage_seq_len).

    Args:
        only_new_tokens: If True, skip generating tokens up to stage_seq_len and only
                        generate tokens beyond that point.
        show_progress: If True, show tqdm progress bar
        stream_tokens: If True, stream tokens to stdout as they're generated

    Returns:
        (generated_text, out_of_bounds_mask, token_ids) where:
        - generated_text: The decoded text
        - out_of_bounds_mask[i] = 1 if token i is out of bounds
        - token_ids: List of token IDs that were generated
    """
    device = compressed_embeddings.device
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id

    batch_size, num_compression_tokens, hidden_size = compressed_embeddings.shape

    generated_token_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
    input_embeddings = model.get_input_embeddings()
    torch_dtype = input_embeddings.weight.dtype

    # If only_new_tokens is True, we only generate new tokens beyond stage_seq_len
    # The compression tokens already represent positions 0 to stage_seq_len-1,
    # so we just generate max_new_tokens directly (starting from position stage_seq_len)
    tokens_to_generate = max_new_tokens

    # Create progress bar
    pbar = None
    if show_progress:
        pbar = tqdm(total=tokens_to_generate, desc="Generating tokens", unit="token", ncols=100)

    for step in range(tokens_to_generate):
        if generated_token_ids.size(1) == 0:
            generated_embeddings = torch.empty(batch_size, 0, hidden_size, device=device)
        else:
            generated_embeddings = input_embeddings(generated_token_ids)
        united_token_embeddings = torch.cat([compressed_embeddings, generated_embeddings], dim=1)
        united_token_embeddings = united_token_embeddings.to(torch_dtype)

        compression_attention_mask = torch.ones((batch_size, num_compression_tokens), dtype=torch.long, device=device)
        attention_mask = torch.ones((batch_size, generated_embeddings.size(1)), dtype=torch.long, device=device)
        united_attention_mask = torch.cat((compression_attention_mask, attention_mask), dim=1)

        outputs = model(inputs_embeds=united_token_embeddings, attention_mask=united_attention_mask)
        logits = outputs.logits[:, -1, :]
        next_token_ids = torch.argmax(logits, dim=-1)

        if eos_token_id is not None:
            if generated_token_ids.size(1) > 0:
                reached_eos = generated_token_ids[:, -1].eq(eos_token_id)
                next_token_ids = torch.where(
                    reached_eos,
                    torch.full_like(next_token_ids, eos_token_id),
                    next_token_ids,
                )

        generated_token_ids = torch.cat([generated_token_ids, next_token_ids.unsqueeze(-1)], dim=-1)

        # Update progress bar and optionally stream token
        if pbar is not None:
            pbar.update(1)
            if stream_tokens:
                token_text = tokenizer.decode([next_token_ids[0].item()], skip_special_tokens=False)
                sys.stdout.write(token_text)
                sys.stdout.flush()

        if eos_token_id is not None and torch.all(next_token_ids.eq(eos_token_id)):
            break

    if pbar is not None:
        pbar.close()
    if stream_tokens:
        sys.stdout.write("\n")
        sys.stdout.flush()

    # If only_new_tokens is True, all generated tokens are new (beyond stage_seq_len)
    # The compression tokens already represent the first stage_seq_len tokens,
    # so all generated tokens are new
    if only_new_tokens and stage_seq_len > 0:
        generated_text = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0]
        token_ids_for_output = generated_token_ids[0].cpu().tolist()
        # All generated tokens are "new" (beyond the compressed sequence)
        num_new_tokens = len(token_ids_for_output)
        out_of_bounds_mask = [1] * num_new_tokens
    else:
        generated_text = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0]
        token_ids_for_output = generated_token_ids[0].cpu().tolist()
        # Determine which tokens are out of bounds
        # Tokens beyond stage_seq_len are out of bounds
        num_generated_tokens = generated_token_ids.shape[1]
        out_of_bounds_mask = [1 if i >= stage_seq_len else 0 for i in range(num_generated_tokens)]

    # Return both the text and the token IDs for accurate highlighting
    return generated_text, out_of_bounds_mask, token_ids_for_output


def highlight_out_of_bounds_tokens(
    tokenizer: AutoTokenizer,
    token_ids: list[int],
    out_of_bounds_mask: list[int],
) -> str:
    """
    Highlight tokens that are out of bounds in red color.

    Args:
        tokenizer: Tokenizer used to decode tokens
        token_ids: List of token IDs from generation
        out_of_bounds_mask: List indicating which token positions are out of bounds

    Returns:
        Text with out-of-bounds tokens highlighted in red
    """
    # Decode each token individually to preserve boundaries
    decoded_tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids]

    # Ensure mask length matches token count
    if len(out_of_bounds_mask) != len(decoded_tokens):
        # If lengths don't match, fall back to decoding all tokens together
        full_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return full_text

    # Build highlighted text
    highlighted_parts = []
    for token_text, is_out_of_bounds in zip(decoded_tokens, out_of_bounds_mask):
        if is_out_of_bounds:
            highlighted_parts.append(f"{RED}{token_text}{RESET}")
        else:
            highlighted_parts.append(token_text)

    return "".join(highlighted_parts)


@torch.no_grad()
def generate_from_text_prefix(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text_prefix: str,
    max_new_tokens: int,
    show_progress: bool = True,
    stream_tokens: bool = False,
) -> tuple[str, list[int]]:
    """
    Generate text from a text prefix using standard greedy decoding (without compression tokens).

    Args:
        model: The language model
        tokenizer: The tokenizer
        text_prefix: The text prefix to generate from
        max_new_tokens: Maximum number of new tokens to generate
        show_progress: If True, show tqdm progress bar
        stream_tokens: If True, stream tokens to stdout as they're generated

    Returns:
        (generated_text, token_ids) where:
        - generated_text: The full generated text (prefix + continuation)
        - token_ids: List of all token IDs (prefix + generated tokens)
    """
    device = next(model.parameters()).device
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id

    # Tokenize the prefix
    prefix_ids = tokenizer.encode(text_prefix, return_tensors="pt", add_special_tokens=True).to(device)
    input_ids = prefix_ids.clone()

    # Create progress bar
    pbar = None
    if show_progress:
        pbar = tqdm(total=max_new_tokens, desc="Generating baseline tokens", unit="token", ncols=100)

    # Generate continuation
    for step in range(max_new_tokens):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1)

        # Handle EOS
        if eos_token_id is not None:
            if input_ids.size(1) > prefix_ids.size(1):
                reached_eos = input_ids[:, -1].eq(eos_token_id)
                next_token_id = torch.where(
                    reached_eos,
                    torch.full_like(next_token_id, eos_token_id),
                    next_token_id,
                )

        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)

        # Update progress bar and optionally stream token
        if pbar is not None:
            pbar.update(1)
            if stream_tokens:
                token_text = tokenizer.decode([next_token_id.item()], skip_special_tokens=False)
                sys.stdout.write(token_text)
                sys.stdout.flush()

        # Stop early if all sequences just produced eos and had eos previously
        if eos_token_id is not None and torch.all(next_token_id.eq(eos_token_id)):
            break

    if pbar is not None:
        pbar.close()
    if stream_tokens:
        sys.stdout.write("\n")
        sys.stdout.flush()

    # Decode the full sequence
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    token_ids = input_ids[0].cpu().tolist()

    return generated_text, token_ids


def load_progressive_artifact(
    artifacts_path: str,
    sample_id: int,
    stage_index: Optional[int] = None,
) -> dict:
    """
    Load a progressive training artifact for a specific sample_id.

    Args:
        artifacts_path: Path to the progressive_prefixes dataset
        sample_id: The sample ID to load
        stage_index: Optional stage index. If None, uses the final stage (highest stage_index)

    Returns:
        Dictionary containing embedding, model_checkpoint, and metadata
    """
    ds = Dataset.load_from_disk(artifacts_path)

    # Filter by sample_id
    matching_rows = []
    for i in range(len(ds)):
        row = ds[i]
        if int(row.get("sample_id", -1)) == int(sample_id):
            matching_rows.append((i, row))

    if not matching_rows:
        raise ValueError(f"No artifacts found for sample_id={sample_id} in '{artifacts_path}'")

    # If stage_index is specified, use it; otherwise use the final stage
    if stage_index is not None:
        for idx, row in matching_rows:
            if int(row.get("stage_index", -1)) == int(stage_index):
                selected_row = row
                break
        else:
            available_stages = [int(r.get("stage_index", -1)) for _, r in matching_rows]
            raise ValueError(
                f"Stage {stage_index} not found for sample_id={sample_id}. " f"Available stages: {sorted(available_stages)}"
            )
    else:
        # Use the final stage (highest stage_index)
        matching_rows.sort(key=lambda x: int(x[1].get("stage_index", 0)))
        selected_row = matching_rows[-1][1]

    embedding = torch.tensor(selected_row["embedding"], dtype=torch.float32)
    model_checkpoint = selected_row.get("model_checkpoint", None)

    if model_checkpoint is None or not model_checkpoint:
        raise ValueError(f"model_checkpoint not found in artifact for sample_id={sample_id}")

    return {
        "embedding": embedding,  # [num_compression_tokens, hidden_size]
        "model_checkpoint": model_checkpoint,
        "sample_id": int(selected_row.get("sample_id", sample_id)),
        "stage_index": int(selected_row.get("stage_index", -1)),
        "stage_seq_len": int(selected_row.get("stage_seq_len", -1)),
        "text": selected_row.get("text", ""),
        "num_compression_tokens": int(selected_row.get("num_compression_tokens", embedding.shape[0])),
        "hidden_size": int(
            selected_row.get("hidden_size", embedding.shape[1] if embedding.dim() == 2 else embedding.shape[-1])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text using greedy decoding from progressive training artifacts")
    parser.add_argument(
        "--artifacts_path",
        type=str,
        required=True,
        help="Path to progressive_prefixes dataset (saved with Dataset.save_to_disk)",
    )
    parser.add_argument(
        "--sample_id",
        type=int,
        required=True,
        help="Sample ID to generate from",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        required=True,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--stage_index",
        type=int,
        default=None,
        help="Optional stage index. If not specified, uses the final stage (highest stage_index)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Optional model checkpoint override. If not specified, uses the checkpoint from the artifact",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save generated text to a file",
    )
    parser.add_argument(
        "--only_new_tokens",
        action="store_true",
        help="Only generate tokens beyond the trained sequence length (skip tokens up to stage_seq_len)",
    )
    parser.add_argument(
        "--no_baseline_generation",
        dest="baseline_generation",
        action="store_false",
        help="Disable baseline generation (default: enabled)",
    )
    parser.set_defaults(baseline_generation=True)
    parser.add_argument(
        "--no_save_to_artifacts",
        dest="save_to_artifacts",
        action="store_false",
        help="Disable saving to artifacts directory (default: enabled)",
    )
    parser.set_defaults(save_to_artifacts=True)
    parser.add_argument(
        "--stream_tokens",
        action="store_true",
        help="Stream tokens to stdout as they're generated (for real-time viewing)",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bars",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load artifact
    print(f"Loading artifact from '{args.artifacts_path}' for sample_id={args.sample_id}...")
    artifact = load_progressive_artifact(
        artifacts_path=args.artifacts_path,
        sample_id=args.sample_id,
        stage_index=args.stage_index,
    )

    embedding = artifact["embedding"]
    model_checkpoint = args.model_checkpoint or artifact["model_checkpoint"]

    print(f"  sample_id: {artifact['sample_id']}")
    print(f"  stage_index: {artifact['stage_index']}")
    print(f"  stage_seq_len: {artifact['stage_seq_len']}")
    print(f"  num_compression_tokens: {artifact['num_compression_tokens']}")
    print(f"  hidden_size: {artifact['hidden_size']}")
    print(f"  model_checkpoint: {model_checkpoint}")

    if artifact["text"]:
        preview = artifact["text"].replace("\n", " ")[:200]
        print(f"  reference text (first 200 chars): {preview}")

    # Load model and tokenizer
    print(f"\nLoading model '{model_checkpoint}'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Prepare embedding tensor
    embedding = embedding.to(device)
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    if embedding.dim() != 2:
        raise ValueError(f"Expected embedding of shape [C, D], got {tuple(embedding.shape)}")

    # Add batch dimension: [1, num_compression_tokens, hidden_size]
    compression_tokens = embedding.unsqueeze(0)

    # Determine artifacts output directory
    artifacts_output_dir = None
    show_progress = not args.no_progress
    if args.save_to_artifacts:
        # Infer output directory from artifacts_path
        # e.g., artifacts/experiments_progressive/exp_name/progressive_prefixes -> artifacts/experiments_progressive/exp_name
        artifacts_path_normalized = os.path.normpath(args.artifacts_path)
        if "progressive_prefixes" in artifacts_path_normalized:
            artifacts_output_dir = os.path.dirname(artifacts_path_normalized)
        elif "compressed_prefixes" in artifacts_path_normalized:
            artifacts_output_dir = os.path.dirname(artifacts_path_normalized)
        else:
            # Fallback: use parent directory
            artifacts_output_dir = os.path.dirname(artifacts_path_normalized)
        os.makedirs(artifacts_output_dir, exist_ok=True)
        print(f"\nWill save results to artifacts directory: {artifacts_output_dir}")

    # Generate with compression tokens
    stage_seq_len = artifact.get("stage_seq_len", -1)
    token_ids = None
    if stage_seq_len <= 0:
        print(f"\nWarning: stage_seq_len={stage_seq_len} is invalid, cannot highlight out-of-bounds tokens")
        print(f"Generating {args.num_tokens} tokens using greedy decoding...")
        generated_texts = generate_from_compression(
            model=model,
            tokenizer=tokenizer,
            compressed_embeddings=compression_tokens,
            max_new_tokens=args.num_tokens,
            num_return_sequences=1,
        )
        generated_text = generated_texts[0]
        highlighted_text = generated_text
    else:
        if args.only_new_tokens:
            print(f"\nGenerating {args.num_tokens} new tokens (skipping first {stage_seq_len} tokens)...")
            print(f"Only tokens beyond position {stage_seq_len} will be generated and shown")
        else:
            print(f"\nGenerating {args.num_tokens} tokens using greedy decoding...")
            print(f"Tokens beyond position {stage_seq_len} will be highlighted in red (out of bounds)")
        generated_text, out_of_bounds_mask, token_ids = generate_with_token_tracking(
            model=model,
            tokenizer=tokenizer,
            compressed_embeddings=compression_tokens,
            max_new_tokens=args.num_tokens,
            stage_seq_len=stage_seq_len,
            only_new_tokens=args.only_new_tokens,
            show_progress=show_progress,
            stream_tokens=args.stream_tokens,
        )
        if args.only_new_tokens:
            # All tokens are new, so all should be highlighted
            highlighted_text = highlight_out_of_bounds_tokens(
                tokenizer=tokenizer,
                token_ids=token_ids,
                out_of_bounds_mask=out_of_bounds_mask,
            )
        else:
            highlighted_text = highlight_out_of_bounds_tokens(
                tokenizer=tokenizer,
                token_ids=token_ids,
                out_of_bounds_mask=out_of_bounds_mask,
            )

    print("\n=== Generated text (with compression tokens) ===")
    if args.only_new_tokens and stage_seq_len > 0:
        print(f"(Only new tokens beyond position {stage_seq_len} are shown, all in red)")
    else:
        print("(Red text indicates tokens beyond the trained sequence length)")
    print(highlighted_text)

    # Generate baseline (without compression tokens) if requested
    baseline_text = None
    baseline_token_ids = None
    if args.baseline_generation:
        if not artifact["text"]:
            print("\nWarning: No reference text found in artifact, cannot generate baseline")
        else:
            if not args.stream_tokens:
                print("\n=== Generating baseline (without compression tokens) ===")
            baseline_text, baseline_token_ids = generate_from_text_prefix(
                model=model,
                tokenizer=tokenizer,
                text_prefix=artifact["text"],
                max_new_tokens=args.num_tokens,
                show_progress=show_progress,
                stream_tokens=args.stream_tokens,
            )
            if not args.stream_tokens:
                print(baseline_text)

    # Save to file if requested
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(f"Sample ID: {artifact['sample_id']}\n")
            f.write(f"Stage Index: {artifact['stage_index']}\n")
            f.write(f"Stage Seq Len: {artifact.get('stage_seq_len', 'N/A')}\n")
            f.write(f"Model: {model_checkpoint}\n")
            f.write(f"Number of tokens generated: {args.num_tokens}\n")
            if args.only_new_tokens:
                f.write(f"Only new tokens mode: True (skipped first {stage_seq_len} tokens)\n")
            f.write(f"\nReference text:\n{artifact['text']}\n")
            f.write(f"\nGenerated text:\n{generated_text}\n")
            if stage_seq_len > 0:
                if args.only_new_tokens:
                    f.write(f"\nNote: Only tokens beyond position {stage_seq_len} were generated (new tokens only).\n")
                else:
                    f.write(
                        f"\nNote: Tokens beyond position {stage_seq_len} are out of bounds of the trained sequence length.\n"
                    )
            if baseline_text:
                f.write("\n=== Baseline generation (without compression tokens) ===\n")
                f.write(f"{baseline_text}\n")
        print(f"\nSaved output to: {os.path.abspath(args.output_file)}")

    # Save to artifacts directory if requested
    if args.save_to_artifacts and artifacts_output_dir:
        results = {
            "sample_id": artifact["sample_id"],
            "stage_index": artifact["stage_index"],
            "stage_seq_len": artifact.get("stage_seq_len", -1),
            "model_checkpoint": model_checkpoint,
            "num_tokens": args.num_tokens,
            "only_new_tokens": args.only_new_tokens,
            "seed": args.seed,
            "reference_text": artifact["text"],
            "generated_with_compression": {
                "text": generated_text,
                "token_ids": token_ids if stage_seq_len > 0 else None,
            },
        }

        if baseline_text:
            results["generated_baseline"] = {
                "text": baseline_text,
                "token_ids": baseline_token_ids,
            }

        # Save JSON file
        output_filename = f"generation_sample_{artifact['sample_id']}_stage_{artifact['stage_index']}.json"
        output_path = os.path.join(artifacts_output_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved generation results to: {output_path}")

        # Also save as readable text files
        compression_output_file = os.path.join(
            artifacts_output_dir,
            f"generation_with_compression_sample_{artifact['sample_id']}_stage_{artifact['stage_index']}.txt",
        )
        with open(compression_output_file, "w", encoding="utf-8") as f:
            f.write(f"Sample ID: {artifact['sample_id']}\n")
            f.write(f"Stage Index: {artifact['stage_index']}\n")
            f.write(f"Stage Seq Len: {artifact.get('stage_seq_len', 'N/A')}\n")
            f.write(f"Model: {model_checkpoint}\n")
            f.write(f"Number of tokens: {args.num_tokens}\n")
            f.write(f"Only new tokens: {args.only_new_tokens}\n")
            f.write(f"\nReference text:\n{artifact['text']}\n")
            f.write(f"\nGenerated text (with compression tokens):\n{generated_text}\n")
        print(f"Saved compression generation to: {compression_output_file}")

        if baseline_text:
            baseline_output_file = os.path.join(
                artifacts_output_dir, f"generation_baseline_sample_{artifact['sample_id']}_stage_{artifact['stage_index']}.txt"
            )
            with open(baseline_output_file, "w", encoding="utf-8") as f:
                f.write(f"Sample ID: {artifact['sample_id']}\n")
                f.write(f"Stage Index: {artifact['stage_index']}\n")
                f.write(f"Model: {model_checkpoint}\n")
                f.write(f"Number of tokens: {args.num_tokens}\n")
                f.write(f"\nReference text (prefix):\n{artifact['text']}\n")
                f.write(f"\nGenerated text (baseline, without compression tokens):\n{baseline_text}\n")
            print(f"Saved baseline generation to: {baseline_output_file}")


if __name__ == "__main__":
    main()
