#!/usr/bin/env python
"""Generate the on-policy dataset and optionally augment it with paraphrases."""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from policy_vector_pipeline import (
    GenerationSettings,
    OnPolicyDataset,
    ReasoningExample,
    ResponseVariant,
    generate_on_policy_variants,
    get_default_device,
    load_dataset,
    load_model_and_tokenizer,
    save_dataset,
)
from policy_vector_pipeline.dataset import (
    build_line_maps,
    create_paraphraser,
    create_similarity,
    load_prompts,
    paraphrase_variants,
    pick_mid_number,
)
from policy_vector_pipeline.paraphrase import Paraphraser

PARAPHRASER_MODELS = [
    "openrouter:openai/gpt-5-mini",
    "openrouter:deepseek/deepseek-r1-0528",
    "openrouter:anthropic/claude-3.5-haiku",
]
DEFAULT_TRAITS = ["evil", "sycophantic", "hallucinating"]


@dataclass(frozen=True)
class ParaphraseTask:
    example_idx: int
    rollout_idx: int
    model_name: str
    edit_scope: str  # "first", "mid", "full"
    line_numbers: Tuple[int, ...]



def generate_base_dataset(args: argparse.Namespace, output_path: Path) -> OnPolicyDataset:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_metadata: Dict[str, object] = {
        "reasoning_model": args.model,
        "paraphraser": None if args.generate_only else args.paraphraser,
        "prompt_source": args.prompt_source,
        "traits": args.traits or [],
        "num_samples": args.num_samples,
    }

    if args.paraphrase_existing:
        if not output_path.exists():
            raise RuntimeError(f"Cannot paraphrase without an existing dataset at {output_path}")
        dataset = load_dataset(output_path)
        dataset.metadata.update({k: v for k, v in run_metadata.items() if v is not None})
    elif args.resume and output_path.exists():
        dataset = load_dataset(output_path)
        dataset.metadata.update({k: v for k, v in run_metadata.items() if v is not None})
        print(f"Resuming from {output_path} with {len(dataset.examples)} existing examples")
    else:
        if output_path.exists():
            raise RuntimeError(
                f"Output file {output_path} already exists. Use --resume or remove the file first."
            )
        dataset = OnPolicyDataset(metadata={k: v for k, v in run_metadata.items() if v is not None})

    existing_prompts = {
        ex.metadata.get("question", ex.prompt)
        for ex in dataset.examples
        if ex.metadata is not None
    }

    if args.paraphrase_existing:
        examples_to_update = [ex for ex in dataset.examples if not ex.off_policy]
        if not examples_to_update:
            print("All examples already have off-policy variants; nothing to paraphrase.")
            return dataset

        paraphraser = create_paraphraser(args.paraphraser, provider=args.paraphraser_provider)
        similarity = create_similarity(args.similarity_model, args.similarity_device)

        updated = 0
        for example in tqdm(examples_to_update, desc="Paraphrasing existing"):
            base_variants = example.on_policy or []
            paired_on, paired_off = paraphrase_variants(
                base_variants,
                paraphraser,
                args.edit_spans,
                similarity,
                args.min_similarity,
            )
            if not paired_on:
                continue
            example.on_policy = paired_on
            example.off_policy = paired_off
            metadata = dict(example.metadata or {})
            metadata.setdefault("question", example.prompt)
            metadata.setdefault(
                "base_on_variants",
                [ResponseVariant(reasoning=v.reasoning, final_answer=v.final_answer).to_dict() for v in base_variants],
            )
            example.metadata = metadata
            updated += 1
            save_dataset(dataset, output_path)

        if updated == 0:
            print("No paraphrases were added. Check paraphraser configuration or similarity threshold.")
        else:
            print(f"Paraphrased {updated} examples. Dataset now has {len(dataset.examples)} examples.")
        return dataset

    traits = args.traits or DEFAULT_TRAITS
    prompts = load_prompts(args.prompt_source, traits)
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]

    if existing_prompts:
        skipped = sum(1 for prompt in prompts if prompt in existing_prompts)
        if skipped:
            print(f"Skipping {skipped} prompts already present in the dataset")
    remaining_prompts = [prompt for prompt in prompts if prompt not in existing_prompts]

    if not remaining_prompts:
        print("No new prompts to process; dataset is up to date.")
        return dataset

    device = get_default_device()
    dtype = torch.float16 if device != "cpu" else torch.float32
    print(f"Loading reasoning model {args.model} on {device} with dtype {dtype}")
    reasoning_model, reasoning_tokenizer = load_model_and_tokenizer(
        args.model,
        device_map=device,
        dtype=dtype,
    )

    paraphraser = None
    similarity = None
    if not args.generate_only:
        paraphraser = create_paraphraser(args.paraphraser, provider=args.paraphraser_provider)
        similarity = create_similarity(args.similarity_model, args.similarity_device)

    gen_settings = GenerationSettings(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )

    total_existing = len(dataset.examples)
    total_final = total_existing + len(remaining_prompts)
    progress_iter = tqdm(remaining_prompts, desc="Prompts")
    current_index = total_existing

    for question in progress_iter:
        current_index += 1
        progress_iter.set_postfix_str(f"processing {current_index}/{total_final}")
        print(
            f"\n[Prompt {current_index}/{total_final}] Generating {args.num_samples} on-policy variants..."
        )
        on_variants = generate_on_policy_variants(
            reasoning_model,
            reasoning_tokenizer,
            question,
            settings=gen_settings,
            num_samples=args.num_samples,
        )
        print(f"Generated {len(on_variants)} variants")

        base_for_metadata = [
            ResponseVariant(reasoning=variant.reasoning, final_answer=variant.final_answer)
            for variant in on_variants
        ]

        if args.generate_only:
            paired_on = [
                ResponseVariant(
                    reasoning=variant.reasoning,
                    final_answer=variant.final_answer,
                    metadata=dict(variant.metadata or {}),
                )
                for variant in on_variants
            ]
            paired_off: List[ResponseVariant] = []
        else:
            paired_on, paired_off = paraphrase_variants(
                on_variants,
                paraphraser,
                args.edit_spans,
                similarity,
                args.min_similarity,
            )

        if not paired_on:
            print("Skipping prompt (no successful paraphrases)")
            continue

        example_id = f"example-{current_index - 1:04d}"
        example_metadata = {
            "question": question,
            "base_on_variants": [variant.to_dict() for variant in base_for_metadata],
        }

        example = ReasoningExample(
            example_id=example_id,
            prompt=question,
            on_policy=paired_on,
            off_policy=paired_off,
            metadata=example_metadata,
        )
        dataset.add(example)
        save_dataset(dataset, output_path)
        print(f"Saved progress: {len(dataset.examples)} examples written to {output_path}")

    print(
        f"Completed {len(dataset.examples) - total_existing} new prompts; dataset now has {len(dataset.examples)} examples."
    )
    return dataset


_PARAPHRASER_CACHE: Dict[str, Paraphraser] = {}


def get_paraphraser(model_name: str) -> Paraphraser:
    if model_name not in _PARAPHRASER_CACHE:
        _PARAPHRASER_CACHE[model_name] = create_paraphraser(model_name)
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
    parser = argparse.ArgumentParser(description="Generate on-policy dataset and augment with paraphrases")
    parser.add_argument(
        "output",
        type=Path,
        help="Output dataset path (augmented dataset when --augment is set)",
    )
    parser.add_argument(
        "--generate-base",
        action="store_true",
        help="Run the base dataset generation stage",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Run the augmentation stage with multi-model paraphrasing",
    )
    parser.add_argument(
        "--base-output",
        type=Path,
        default=None,
        help="Where to save the base dataset when both stages run (default: <output>_base.json)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Existing dataset to augment when skipping base generation",
    )

    base = parser.add_argument_group("Base generation")
    base.add_argument("--model", default="Qwen/Qwen3-4B", help="Reasoning model for on-policy rollouts")
    base.add_argument(
        "--paraphraser",
        default="openrouter:gpt-oss-20b",
        help="Model used to paraphrase on-policy variants for off-policy counterparts",
    )
    base.add_argument(
        "--paraphraser-provider",
        default="deepinfra",
        help="Optional OpenRouter provider name when using openrouter paraphraser",
    )
    base.add_argument("--prompt-source", default="persona_extract", help="Prompt source identifier")
    base.add_argument("--traits", nargs="*", help="Traits to pull prompts from (persona sources only)")
    base.add_argument("--num-samples", type=int, default=5, help="On-policy samples per prompt")
    base.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Limit number of prompts processed in this run",
    )
    base.add_argument(
        "--edit-spans",
        nargs="*",
        type=int,
        default=[1],
        help="Number of leading reasoning lines to rewrite for off-policy variants",
    )
    base.add_argument("--temperature", type=float, default=0.6)
    base.add_argument("--top-p", type=float, default=0.95)
    base.add_argument("--top-k", type=int, default=20)
    base.add_argument("--max-new-tokens", type=int, default=8096)
    base.add_argument("--similarity-model", default="Qwen/Qwen3-Embedding-0.6B")
    base.add_argument("--similarity-device", default=None)
    base.add_argument("--min-similarity", type=float, default=0.8)
    base.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate on-policy traces without creating paraphrased variants",
    )
    base.add_argument("--paraphrase-existing", action="store_true", help="Rewrite missing off-policy entries in place")
    base.add_argument("--resume", action="store_true", help="Resume generation when the output already exists")

    aug = parser.add_argument_group("Augmentation")
    aug.add_argument(
        "--augment-prompts",
        type=int,
        default=None,
        help="Limit augmentation to the first N prompts",
    )
    aug.add_argument(
        "--rollouts-per-prompt",
        type=int,
        default=2,
        help="Number of on-policy rollouts per prompt to paraphrase during augmentation",
    )
    aug.add_argument("--max-workers", type=int, default=10, help="Thread pool size for augmentation")
    aug.add_argument(
        "--augment-models",
        nargs="*",
        default=PARAPHRASER_MODELS,
        help="List of OpenRouter model identifiers for augmentation",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.generate_base and not args.augment:
        raise SystemExit("Nothing to do: enable --generate-base, --augment, or both")

    dataset: Optional[OnPolicyDataset] = None
    base_path: Optional[Path] = None

    if args.generate_base:
        if args.augment:
            base_path = args.base_output or args.output.with_name(args.output.stem + "_base.json")
            if args.base_output is None:
                print(f"Base dataset will be saved to {base_path}")
        else:
            base_path = args.base_output or args.output
        dataset = generate_base_dataset(args, base_path)
        if not args.augment and base_path != args.output:
            save_dataset(dataset, args.output)
            print(f"Base dataset copied to {args.output}")

    if args.augment:
        if dataset is None:
            load_path = args.input or args.output
            if not load_path.exists():
                raise RuntimeError(f"Cannot augment without a dataset at {load_path}")
            dataset = load_dataset(load_path)
        start = time.time()
        augmented = augment_dataset(
            dataset,
            num_prompts=args.augment_prompts,
            rollouts_per_prompt=args.rollouts_per_prompt,
            models=args.augment_models,
            max_workers=args.max_workers,
        )
        duration = time.time() - start
        save_dataset(augmented, args.output)
        summarize(augmented)
        print(f"Augmented dataset saved to {args.output}")
        print(f"Elapsed time: {duration/60:.2f} minutes")
    elif dataset is not None:
        summarize(dataset)


if __name__ == "__main__":
    main()
