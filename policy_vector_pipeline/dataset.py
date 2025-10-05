"""Helpers for dataset construction and paraphrasing workflows."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .types import ResponseVariant
from .paraphrase import Paraphraser
from .utils import EmbeddingSimilarity, get_default_device

THINK_TAGS = {"<think>", "</think>"}


def load_prompts(source: str, traits: Sequence[str], *, prompts_root: Optional[Path] = None) -> List[str]:
    """Load unique prompt questions for a given trait source."""
    root = prompts_root or Path(__file__).resolve().parents[1] / "prompts"
    dataset = "trait_data_extract" if source == "persona_extract" else "trait_data_eval"

    prompts: List[str] = []
    seen: set[str] = set()

    for trait in traits:
        path = root / dataset / f"{trait}.json"
        if not path.exists():
            raise FileNotFoundError(f"Trait file not found: {path}")
        entries = json.loads(path.read_text())
        for question in entries.get("questions", []):
            if question not in seen:
                prompts.append(question)
                seen.add(question)
    return prompts


def count_nonempty_reasoning_lines(reasoning: str) -> int:
    return sum(
        1
        for line in reasoning.splitlines()
        if (stripped := line.strip()) and stripped not in THINK_TAGS
    )


def content_indices_for_line_numbers(line_numbers: Sequence[int]) -> List[int]:
    return [max(0, num - 1) for num in line_numbers]


def clone_variant_with_metadata(variant: ResponseVariant, metadata: Dict[str, object]) -> ResponseVariant:
    merged = dict(variant.metadata or {})
    merged.update(metadata)
    return ResponseVariant(
        reasoning=variant.reasoning,
        final_answer=variant.final_answer,
        metadata=merged,
    )


def paraphrase_variants(
    base_variants: Sequence[ResponseVariant],
    paraphraser: Paraphraser,
    edit_spans: Sequence[int],
    similarity: Optional[EmbeddingSimilarity],
    min_similarity: float,
) -> Tuple[List[ResponseVariant], List[ResponseVariant]]:
    """Paraphrase leading reasoning spans to produce on/off-policy pairs."""
    paired_on: List[ResponseVariant] = []
    paired_off: List[ResponseVariant] = []

    for variant in base_variants:
        available_lines = count_nonempty_reasoning_lines(variant.reasoning)
        if available_lines == 0:
            continue
        for span in edit_spans:
            if span <= 0 or available_lines < span:
                continue
            rewritten = paraphraser.rewrite(variant.reasoning, span)
            if rewritten is None:
                continue

            similarity_score: Optional[float] = None
            if similarity is not None:
                similarity_score = similarity.similarity(
                    variant.compose(include_answer=False),
                    ResponseVariant(rewritten, variant.final_answer).compose(include_answer=False),
                )
                if similarity_score < min_similarity:
                    continue

            line_numbers = list(range(1, span + 1))
            content_indices = content_indices_for_line_numbers(line_numbers)

            on_metadata: Dict[str, object] = {
                "paired_edit_span": span,
                "paired_content_indices": content_indices,
                "paired_line_numbers": line_numbers,
            }
            if similarity_score is not None:
                on_metadata["paired_similarity"] = similarity_score

            off_metadata: Dict[str, object] = {
                "edit_type": "paraphrase",
                "edit_span": span,
                "edited_content_indices": content_indices,
                "edited_line_numbers": line_numbers,
            }
            if similarity_score is not None:
                off_metadata["similarity"] = similarity_score

            paired_on.append(clone_variant_with_metadata(variant, on_metadata))
            paired_off.append(
                ResponseVariant(
                    reasoning=rewritten,
                    final_answer=variant.final_answer,
                    metadata=off_metadata,
                )
            )
    return paired_on, paired_off


def build_line_maps(reasoning: str) -> Tuple[Dict[int, int], Dict[int, int]]:
    number_to_raw: Dict[int, int] = {}
    raw_to_number: Dict[int, int] = {}
    counter = 0
    for idx, line in enumerate(reasoning.splitlines()):
        stripped = line.strip()
        if not stripped or stripped in THINK_TAGS:
            continue
        counter += 1
        number_to_raw[counter] = idx
        raw_to_number[idx] = counter
    return number_to_raw, raw_to_number


def pick_mid_number(number_to_raw: Dict[int, int]) -> Optional[int]:
    if len(number_to_raw) <= 2:
        return None
    return (len(number_to_raw) + 1) // 2


def create_paraphraser(
    model_name: str,
    *,
    provider: Optional[str] = None,
    requests_per_minute: int = 100,
    device_preference: Optional[str] = None,
) -> Paraphraser:
    return Paraphraser(
        model_name,
        provider=provider,
        requests_per_minute=requests_per_minute,
        device_preference=device_preference or get_default_device(),
    )


def create_similarity(model_name: str, device: Optional[str] = None) -> Optional[EmbeddingSimilarity]:
    if model_name.lower() == "none":
        return None
    return EmbeddingSimilarity(model_name, device=device or get_default_device())
