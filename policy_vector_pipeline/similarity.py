from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer, util


@dataclass
class EmbeddingSimilarity:
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = "cpu"
    prompt_name: str = "document"

    def __post_init__(self) -> None:
        if self.device == "auto" or self.device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
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
