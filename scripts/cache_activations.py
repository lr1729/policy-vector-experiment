#!/usr/bin/env python
"""Cache activations to avoid re-collecting for analysis."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(candidate) not in sys.path and candidate.exists():
        sys.path.append(str(candidate))

from policy_vector_pipeline import (
    ActivationCollector,
    load_dataset,
    load_model_and_tokenizer,
)


def main():
    parser = argparse.ArgumentParser(description="Cache activations for a dataset")
    parser.add_argument("dataset", type=Path, help="Dataset JSON path")
    parser.add_argument("output", type=Path, help="Output pickle file")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Model name")
    parser.add_argument("--layers", nargs="*", type=int, help="Layers to extract")
    parser.add_argument("--reduction", default="mean", choices=["mean", "last"])
    parser.add_argument("--include-answer", action="store_true")
    parser.add_argument("--clip-to-span", action="store_true")

    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    model, tokenizer = load_model_and_tokenizer(args.model)

    collector = ActivationCollector(
        model,
        tokenizer,
        layers=args.layers,
        reduction=args.reduction,
        response_only=True,
        include_answer=args.include_answer,
    )

    acts = collector.collect_dataset(
        dataset,
        progress=True,
        clip_to_span=args.clip_to_span,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(acts, f)

    print(f"Cached activations to {args.output}")
    print(f"  On-policy: {len(acts['on'][list(acts['on'].keys())[0]])} samples")
    print(f"  Off-policy: {len(acts['off'][list(acts['off'].keys())[0]])} samples")


if __name__ == "__main__":
    main()
