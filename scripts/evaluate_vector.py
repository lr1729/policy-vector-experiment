#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from policy_vector_pipeline import ActivationCollector, MeanDifferenceVector, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a mean-difference vector on a dataset")
    parser.add_argument("dataset", type=Path, help="Path to on/off-policy dataset JSON")
    parser.add_argument("vector", type=Path, help="Path to saved mean-difference vector (.pt)")
    parser.add_argument("--model", default=None, help="Model name (defaults to dataset metadata)")
    parser.add_argument("--reduction", default="mean", choices=["mean", "last"], help="Token reduction strategy")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples for quick checks")
    parser.add_argument("--device-map", default="auto", help="transformers device_map argument")
    parser.add_argument("--dtype", default="auto", help="Ignored if running on CPU; passed to from_pretrained")
    parser.add_argument("--top", type=int, default=5, help="Show top-N layers by Cohen d")
    return parser.parse_args()


def load_model(model_name: str, device_map: str, dtype: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def compute_stats(acts: dict[int, list[torch.Tensor]], vector: MeanDifferenceVector):
    stats = []
    for layer, vec in vector.layer_vectors.items():
        on_stack = torch.stack(acts["on"][layer]).to(torch.float32)
        off_stack = torch.stack(acts["off"][layer]).to(torch.float32)
        vec = vec.to(torch.float32)
        vec /= torch.linalg.norm(vec) + 1e-8
        proj_on = (on_stack @ vec).cpu()
        proj_off = (off_stack @ vec).cpu()
        mean_on = proj_on.mean().item()
        mean_off = proj_off.mean().item()
        diff = mean_on - mean_off
        var_on = proj_on.var(unbiased=False).item()
        var_off = proj_off.var(unbiased=False).item()
        pooled = torch.sqrt((proj_on.var(unbiased=False) + proj_off.var(unbiased=False)) / 2).item()
        cohens_d = diff / (pooled + 1e-8)
        threshold = (mean_on + mean_off) / 2
        on_acc = (proj_on > threshold).float().mean().item()
        off_acc = (proj_off <= threshold).float().mean().item()
        stats.append(
            {
                "layer": layer,
                "mean_on": mean_on,
                "mean_off": mean_off,
                "diff": diff,
                "std_on": var_on ** 0.5,
                "std_off": var_off ** 0.5,
                "cohens_d": cohens_d,
                "threshold": threshold,
                "on_acc": on_acc,
                "off_acc": off_acc,
                "overall_acc": (on_acc + off_acc) / 2,
            }
        )
    stats.sort(key=lambda item: item["cohens_d"], reverse=True)
    return stats


def main() -> None:
    args = parse_args()

    dataset = load_dataset(args.dataset)
    model_name = args.model or dataset.metadata.get("reasoning_model")
    if not model_name:
        raise ValueError("Model name missing; pass --model or populate dataset metadata")

    vector = MeanDifferenceVector.load(args.vector)
    layer_ids = sorted(vector.layer_vectors.keys())

    model, tokenizer = load_model(model_name, args.device_map, args.dtype)

    collector = ActivationCollector(
        model,
        tokenizer,
        layers=layer_ids,
        reduction=args.reduction,
        response_only=True,
    )

    acts = collector.collect_dataset(dataset, progress=True, limit=args.limit)

    stats = compute_stats(acts, vector)

    print(f"\nTop {min(args.top, len(stats))} layers by Cohen d:")
    for entry in stats[: args.top]:
        print(
            f"Layer {entry['layer']:>2}: diff={entry['diff']:.3f}, d={entry['cohens_d']:.3f}, "
            f"overall_acc={entry['overall_acc']*100:.1f}%, on_acc={entry['on_acc']*100:.1f}%, "
            f"off_acc={entry['off_acc']*100:.1f}%"
        )

    means = [entry["cohens_d"] for entry in stats]
    print(f"\nLayer stats summary: mean d={mean(means):.3f}, max d={max(means):.3f}, min d={min(means):.3f}")

    best = stats[0]
    layer = best["layer"]
    vec = vector.layer_vectors[layer].to(torch.float32)
    vec /= torch.linalg.norm(vec) + 1e-8
    proj_on = torch.stack(acts["on"][layer]).to(torch.float32) @ vec
    proj_off = torch.stack(acts["off"][layer]).to(torch.float32) @ vec
    print(
        f"\nBest layer {layer}: on mean={best['mean_on']:.3f} ± {best['std_on']:.3f}, "
        f"off mean={best['mean_off']:.3f} ± {best['std_off']:.3f}, threshold={best['threshold']:.3f}"
    )
    print(
        f"Projection ranges -> on [{proj_on.min():.3f}, {proj_on.max():.3f}], "
        f"off [{proj_off.min():.3f}, {proj_off.max():.3f}]"
    )


if __name__ == "__main__":
    main()
