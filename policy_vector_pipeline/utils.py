from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def get_default_device() -> str:
    """Detect best available device (mps > cuda > cpu)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(
    model_name: str,
    *,
    device_map: Optional[str] = "auto",
    dtype: Optional[str | torch.dtype] = "auto",
    trust_remote_code: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a model and tokenizer with sensible defaults.

    Args:
        model_name: HuggingFace model identifier
        device_map: Device map for model loading (default: "auto")
        dtype: Data type for model weights (default: "auto")
        trust_remote_code: Whether to trust remote code (default: True)

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# Similarity utilities (previously in similarity.py)
class EmbeddingSimilarity:
    """Compute semantic similarity using sentence embeddings."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        prompt_name: str = "document",
    ):
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.device = device or get_default_device()

        model_kwargs = {}
        tokenizer_kwargs = {"padding_side": "left"}
        self.model = SentenceTransformer(
            self.model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            device=self.device,
        )

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        return self.model.encode(
            texts,
            batch_size=4,
            convert_to_tensor=True,
            prompt_name=self.prompt_name,
            normalize_embeddings=True,
        )

    def pairwise(self, texts_a: Sequence[str], texts_b: Sequence[str]) -> torch.Tensor:
        if len(texts_a) != len(texts_b):
            raise ValueError("texts_a and texts_b must have the same length")
        embeddings_a = self.encode(texts_a)
        embeddings_b = self.encode(texts_b)
        scores = util.cos_sim(embeddings_a, embeddings_b)
        return scores.diag()

    def similarity(self, text_a: str, text_b: str) -> float:
        score = self.pairwise([text_a], [text_b])
        return float(score.item())
