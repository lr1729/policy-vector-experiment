from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set


PERSONA_DATA_DIR = Path("persona_vectors") / "data_generation"


def _load_persona_file(path: Path) -> List[str]:
    data = json.loads(path.read_text())
    return data.get("questions", [])


def load_persona_prompts(
    traits: Sequence[str],
    *,
    dataset: str = "trait_data_extract",
    base_dir: Path = PERSONA_DATA_DIR,
) -> List[str]:
    """Load prompt questions from persona_vectors trait data.

    Args:
        traits: Trait names (e.g., ["evil", "sycophancy"]).
        dataset: One of the subdirectories under persona_vectors/data_generation.
        base_dir: Root directory of persona_vectors data.

    Returns:
        Deduplicated list of question strings.
    """
    prompts: List[str] = []
    seen: Set[str] = set()
    root = base_dir / dataset
    for trait in traits:
        path = root / f"{trait}.json"
        if not path.exists():
            raise FileNotFoundError(f"Persona trait file not found: {path}")
        for question in _load_persona_file(path):
            if question not in seen:
                prompts.append(question)
                seen.add(question)
    return prompts


def load_prompts(source: str, *, traits: Optional[Sequence[str]] = None) -> List[str]:
    """High-level helper for fetching prompt lists.

    Currently supports:
        - "persona_extract"
        - "persona_eval"

    Args:
        source: Identifier for the prompt source.
        traits: Optional trait names when using persona sources. If omitted, defaults
            to the canonical trio from the Persona Vectors paper.
    """
    if source not in {"persona_extract", "persona_eval"}:
        raise ValueError(f"Unsupported prompt source '{source}'")

    default_traits = ("evil", "sycophancy", "hallucination")
    trait_list = list(traits) if traits else list(default_traits)
    dataset = "trait_data_extract" if source == "persona_extract" else "trait_data_eval"
    return load_persona_prompts(trait_list, dataset=dataset)
