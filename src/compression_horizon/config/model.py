"""Model-loading arguments shared across training, evaluation and analysis scripts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelArgs:
    """How to load the frozen language model that hosts compression embeddings.

    Reused by every script that loads a HuggingFace causal LM (training,
    HellaSwag/ARC/MMLU eval, attention-mass analysis, generation, etc.).
    """

    model_checkpoint: str = field(
        default="HuggingFaceTB/SmolLM2-135M",
        metadata={"help": "HuggingFace location for a model and a tokenizer."},
    )
    dtype: str = field(
        default="bf16",
        metadata={
            "help": (
                "Torch dtype for model and training. "
                "One of: auto, float32|fp32, bfloat16|bf16, float16|fp16. "
                "This overrides the torch_dtype used to load the model."
            )
        },
    )
    attn_implementation: str | None = field(
        default="flash_attention_2",
        metadata={
            "help": (
                "Attention implementation passed to AutoModelForCausalLM.from_pretrained. "
                "Common values: 'flash_attention_2', 'sdpa', 'eager'. "
                "Set to 'eager' on CPU or when flash-attn is not installed."
            )
        },
    )
