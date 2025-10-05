#!/usr/bin/env python
"""Augment the on-policy dataset with multi-model paraphrases."""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from policy_vector_pipeline import (
    OnPolicyDataset,
    ReasoningExample,
    ResponseVariant,
    load_dataset,
    save_dataset,
)
from policy_vector_pipeline.paraphrase import Paraphraser

PARAPHRASER_MODELS = [
    "openrouter:openai/gpt-5-mini",
    "openrouter:deepseek/deepseek-r1-0528",
    "openrouter:anthropic/claude-3.5-haiku",
]


@dataclass(frozen=True)
class ParaphraseTask:
    example_idx: int
    rollout_idx: int
    model_name: str
    edit_scope: str  # "first", "mid", "full"
    line_numbers: Tuple[int, ...]


def build_line_maps(reasoning: str) -> Tuple[Dict[int, int], Dict[int, int]]:
    number_to_raw: Dict[int, int] = {}
    raw_to_number: Dict[int, int] = {}
    counter = 0
    for idx, line in enumerate(reasoning.splitlines()):
        stripped = line.strip()
        if not stripped or stripped in {"<think>", "</think>"}:
            continue
        counter += 1
        number_to_raw[counter] = idx
        raw_to_number[idx] = counter
    return number_to_raw, raw_to_number


def pick_mid_number(number_to_raw: Dict[int, int]) -> int | None:
    if len(number_to_raw) <= 2:
        return None
    return (len(number_to_raw) + 1) // 2


# Simple cache of paraphraser instances per model to reuse RolloutsClient connections
_PARAPHRASER_CACHE: Dict[str, Paraphraser] = {}


def get_paraphraser(model_name: str) -> Paraphraser:
    if model_name not in _PARAPHRASER_CACHE:
        _PARAPHRASER_CACHE[model_name] = Paraphraser(model_name, requests_per_minute=100)
    return _PARAPHRASER_CACHE[model_name]


def run_paraphrase_task(dataset: OnPolicyDataset, task: ParaphraseTask) -> Tuple[int, ResponseVariant] | None:
    example = dataset.examples[task.example_idx]
    base = example.on_policy[task.rollout_idx]
    paraphraser = get_paraphraser(task.model_name)
    rewritten = paraphraser.rewrite(base.reasoning, list(task.line_numbers))
    if rewritten is None:
        return None

    metadata = dict(base.metadata or {})
    line_numbers = list(task.line_numbers)
    metadata.update(
        {
            "paraphraser": task.model_name,
            "edit_scope": task.edit_scope,
            "line_numbers": line_numbers,
            "paired_line_numbers": line_numbers,
            "source_rollout": task.rollout_idx,
        }
    )
    variant = ResponseVariant(
        reasoning=rewritten,
        final_answer=base.final_answer,
        metadata=metadata,
    )
    return task.example_idx, variant


def generate_tasks(
    dataset: OnPolicyDataset,
    *,
    num_prompts: int | None,
    rollouts_per_prompt: int,
    models: Sequence[str],
) -> Tuple[List[ParaphraseTask], List[int]]:
    total_examples = len(dataset.examples)
    if num_prompts is not None:
        example_indices = list(range(min(num_prompts, total_examples)))
    else:
        example_indices = list(range(total_examples))

    tasks: List[ParaphraseTask] = []
    usable_indices: List[int] = []

    for ex_idx in example_indices:
        example = dataset.examples[ex_idx]
        if not example.on_policy:
            continue
        usable_indices.append(ex_idx)
        rollouts = example.on_policy[:rollouts_per_prompt]
        for rollout_idx, base in enumerate(rollouts):
            number_to_raw, _ = build_line_maps(base.reasoning)
            if not number_to_raw:
                continue
            first_numbers = (1,)
            mid_number = pick_mid_number(number_to_raw)
            full_numbers = tuple(range(1, len(number_to_raw) + 1))
            for model_name in models:
                tasks.append(
                    ParaphraseTask(
                        example_idx=ex_idx,
                        rollout_idx=rollout_idx,
                        model_name=model_name,
                        edit_scope="first",
                        line_numbers=first_numbers,
                    )
                )
                if mid_number is not None:
                    tasks.append(
                        ParaphraseTask(
                            example_idx=ex_idx,
                            rollout_idx=rollout_idx,
                            model_name=model_name,
                            edit_scope="mid",
                            line_numbers=(mid_number,),
                        )
                    )
                tasks.append(
                    ParaphraseTask(
                        example_idx=ex_idx,
                        rollout_idx=rollout_idx,
                        model_name=model_name,
                        edit_scope="full",
                        line_numbers=full_numbers,
                    )
                )
    return tasks, usable_indices


def augment_dataset(
    dataset: OnPolicyDataset,
    *,
    num_prompts: int | None,
    rollouts_per_prompt: int,
    models: Sequence[str],
    max_workers: int,
) -> OnPolicyDataset:
    tasks, usable_indices = generate_tasks(
        dataset,
        num_prompts=num_prompts,
        rollouts_per_prompt=rollouts_per_prompt,
        models=models,
    )

    new_variants: Dict[int, List[ResponseVariant]] = {idx: [] for idx in usable_indices}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_paraphrase_task, dataset, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            example_idx, variant = result
            new_variants.setdefault(example_idx, []).append(variant)

    augmented_examples: List[ReasoningExample] = []
    for idx, example in enumerate(dataset.examples):
        merged_off_policy = list(example.off_policy or [])
        merged_off_policy.extend(new_variants.get(idx, []))
        augmented_examples.append(
            ReasoningExample(
                example_id=example.example_id,
                prompt=example.prompt,
                on_policy=example.on_policy,
                off_policy=merged_off_policy,
                correct_answer=example.correct_answer,
                metadata=example.metadata,
            )
        )

    return OnPolicyDataset(examples=augmented_examples, metadata=dataset.metadata)


def summarize(dataset: OnPolicyDataset) -> None:
    total_on = sum(len(ex.on_policy or []) for ex in dataset.examples)
    total_off = sum(len(ex.off_policy or []) for ex in dataset.examples)
    print(f"Summary: prompts={len(dataset.examples)}, on_policy={total_on}, off_policy={total_off}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment dataset with paraphrased variants")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/on_policy_persona.json"),
        help="Input dataset path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/on_policy_persona_augmented.json"),
        help="Output dataset path",
    )
    parser.add_argument(
        "--prompts",
        type=int,
        default=None,
        help="Limit to first N prompts (default: all)",
    )
    parser.add_argument(
        "--rollouts-per-prompt",
        type=int,
        default=2,
        help="Number of on-policy rollouts per prompt to paraphrase",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Thread pool size for concurrent paraphrasing",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=PARAPHRASER_MODELS,
        help="List of OpenRouter model identifiers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    start = time.time()
    augmented = augment_dataset(
        dataset,
        num_prompts=args.prompts,
        rollouts_per_prompt=args.rollouts_per_prompt,
        models=args.models,
        max_workers=args.max_workers,
    )
    duration = time.time() - start
    save_dataset(augmented, args.output)
    print(f"Augmented dataset saved to {args.output}")
    summarize(augmented)
    print(f"Elapsed time: {duration/60:.2f} minutes")


if __name__ == "__main__":
    main()
