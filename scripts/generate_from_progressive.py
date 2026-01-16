"""Generate text using greedy decoding from progressive training artifacts."""

import argparse
import os
from typing import Optional

import torch
from datasets import Dataset
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
) -> tuple[str, list[int], list[int]]:
    """
    Generate text and return both the text and a list indicating which token positions
    are out of bounds (beyond stage_seq_len).

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

    for _ in range(max_new_tokens):
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

        if eos_token_id is not None and torch.all(next_token_ids.eq(eos_token_id)):
            break

    generated_text = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0]

    # Determine which tokens are out of bounds
    # Tokens beyond stage_seq_len are out of bounds
    num_generated_tokens = generated_token_ids.shape[1]
    out_of_bounds_mask = [1 if i >= stage_seq_len else 0 for i in range(num_generated_tokens)]

    # Return both the text and the token IDs for accurate highlighting
    return generated_text, out_of_bounds_mask, generated_token_ids[0].cpu().tolist()


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

    # Generate with token tracking
    stage_seq_len = artifact.get("stage_seq_len", -1)
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
        print(f"\nGenerating {args.num_tokens} tokens using greedy decoding...")
        print(f"Tokens beyond position {stage_seq_len} will be highlighted in red (out of bounds)")
        generated_text, out_of_bounds_mask, token_ids = generate_with_token_tracking(
            model=model,
            tokenizer=tokenizer,
            compressed_embeddings=compression_tokens,
            max_new_tokens=args.num_tokens,
            stage_seq_len=stage_seq_len,
        )
        highlighted_text = highlight_out_of_bounds_tokens(
            tokenizer=tokenizer,
            token_ids=token_ids,
            out_of_bounds_mask=out_of_bounds_mask,
        )

    print("\n=== Generated text ===")
    print("(Red text indicates tokens beyond the trained sequence length)")
    print(highlighted_text)

    # Save to file if requested
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(f"Sample ID: {artifact['sample_id']}\n")
            f.write(f"Stage Index: {artifact['stage_index']}\n")
            f.write(f"Stage Seq Len: {artifact.get('stage_seq_len', 'N/A')}\n")
            f.write(f"Model: {model_checkpoint}\n")
            f.write(f"Number of tokens generated: {args.num_tokens}\n")
            f.write(f"\nReference text:\n{artifact['text']}\n")
            f.write(f"\nGenerated text:\n{generated_text}\n")
            if stage_seq_len > 0:
                f.write(f"\nNote: Tokens beyond position {stage_seq_len} are out of bounds of the trained sequence length.\n")
        print(f"\nSaved output to: {os.path.abspath(args.output_file)}")


if __name__ == "__main__":
    main()
