#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(candidate) not in sys.path and candidate.exists():
        sys.path.append(str(candidate))

from transformers import AutoModelForCausalLM, AutoTokenizer

from policy_vector_pipeline import ActivationCollector, MeanDifferenceVector
from policy_vector_pipeline.io import load_dataset


def load_reasoning_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract mean-difference vector from dataset")
    parser.add_argument("dataset", type=Path, help="Path to dataset JSON")
    parser.add_argument("output", type=Path, help="Path to save vector (pt file)")
    parser.add_argument("--model", help="Override reasoning model (defaults to dataset metadata)")
    parser.add_argument("--layers", nargs="*", type=int, help="Specific layers to extract (default mid-to-late)")
    parser.add_argument("--reduction", default="mean", choices=["mean", "last"], help="Token reduction strategy")

    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    model_name = args.model or dataset.metadata.get("reasoning_model")
    if not model_name:
        raise ValueError("Model name must be provided either via --model or dataset metadata")

    model, tokenizer = load_reasoning_model(model_name)
    layers = args.layers if args.layers else None

    collector = ActivationCollector(
        model,
        tokenizer,
        layers=layers,
        reduction=args.reduction,
        response_only=True,
    )

    activations = collector.collect_dataset(dataset)
    vector = MeanDifferenceVector.from_activations(activations["on"], activations["off"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vector.save(args.output)
    print(f"Saved vector with {len(vector.layer_vectors)} layers to {args.output}")


if __name__ == "__main__":
    main()
