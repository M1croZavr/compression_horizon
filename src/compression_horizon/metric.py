from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast


def calculate_distances(compression_embeddings: torch.Tensor, sequence_embeddings: torch.Tensor) -> tuple[float, float, float]:
    # Cosine
    cosine = F.cosine_similarity(compression_embeddings, sequence_embeddings, dim=-1)
    cosine = (1.0 - cosine).mean().item()
    # l2
    l2 = torch.sqrt(torch.sum((sequence_embeddings - compression_embeddings) ** 2, dim=-1)).mean().item()
    # l1
    l1 = torch.sum(torch.abs(sequence_embeddings - compression_embeddings), dim=-1).mean().item()
    return cosine, l2, l1


@torch.no_grad()
def calculate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer,
    compressed_embeddings: torch.Tensor,  # [1, mem, hidden]
    sequence_embeddings: torch.Tensor,  # [1, sequence, hidden]
    attention_mask: torch.Tensor,  # [1, sequence]
    *,
    n: int = 128,
) -> float:
    """Entropy measures the level of uncertainty in the model's output.
    Lower entropy means the model is more certain about its predictions and therefore, the perplexity is lower.
    Perplexity indicates the level of confidence the model has in its prediction—lower perplexity suggests higher
    confidence and better performance in predicting the next word,
    while higher perplexity signals more uncertainty and less reliability."""
    # Cast to the same device
    device = compressed_embeddings.device
    if model.device != device:
        model = model.to(device)
    model.eval()

    # Add pad_token to a tokenizer
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    eos_token_id = tokenizer.eos_token_id

    _, num_compression_tokens, _ = compressed_embeddings.shape

    # Container for logits
    generated_token_logits = list()
    # Model's input embeddings layer
    input_embeddings = model.get_input_embeddings()
    torch_dtype = input_embeddings.weight.dtype

    for _ in range(n):
        # Embeddings
        united_token_embeddings = torch.cat((compressed_embeddings, sequence_embeddings), dim=1)  # [1, mem + sequence, hidden]
        united_token_embeddings = united_token_embeddings.to(torch_dtype)

        # Attention mask
        compression_attention_mask = torch.ones((1, num_compression_tokens), dtype=torch.long, device=device)  # [1, mem]
        united_attention_mask = torch.cat((compression_attention_mask, attention_mask), dim=1)  # [1, mem + sequence]

        outputs = model(inputs_embeds=united_token_embeddings, attention_mask=united_attention_mask)
        logits = outputs.logits[:, -1, :]  # [1, vocabulary]
        next_token_id = torch.argmax(logits, dim=-1)  # [1]

        # Stop if a sequence already reached EOS token
        if eos_token_id is not None:
            if next_token_id.item() == eos_token_id:
                break
        generated_token_logits.append(logits)
        # Increment sequence embeddings and attention mask
        next_token_embedding = input_embeddings(next_token_id).unsqueeze(dim=1)  # [1, 1, hidden]
        sequence_embeddings = torch.cat((sequence_embeddings, next_token_embedding), dim=1)  # [1, sequence + 1, hidden]
        attention_mask = torch.cat(
            (attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)), dim=1
        )  # [1, sequence + 1]

    generated_token_logits = torch.cat(generated_token_logits, dim=0)
    generated_token_log_probs = F.log_softmax(generated_token_logits, dim=1)
    cross_entropy = -1 * generated_token_log_probs[generated_token_log_probs.argmax(dim=1).view(-1, 1)].mean()
    perplexity = torch.exp(cross_entropy).item()
    return perplexity
