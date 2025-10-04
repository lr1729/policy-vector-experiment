#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(candidate) not in sys.path and candidate.exists():
        sys.path.append(str(candidate))

from tqdm.auto import tqdm

from policy_vector_pipeline import (
    GenerationSettings,
    OnPolicyDataset,
    ReasoningExample,
    ResponseVariant,
    generate_on_policy_variants,
    get_default_device,
    load_model_and_tokenizer,
    load_dataset,
    save_dataset,
    EmbeddingSimilarity,
)
from policy_vector_pipeline.paraphrase import Paraphraser


def _load_prompts(source: str, traits: Sequence[str]) -> List[str]:
    """Load prompts from trait JSON files."""
    prompts_dir = PROJECT_ROOT / "prompts"
    dataset = "trait_data_extract" if source == "persona_extract" else "trait_data_eval"

    prompts: List[str] = []
    seen: Set[str] = set()

    for trait in traits:
        path = prompts_dir / dataset / f"{trait}.json"
        if not path.exists():
            raise FileNotFoundError(f"Trait file not found: {path}")

        data = json.loads(path.read_text())
        for question in data.get("questions", []):
            if question not in seen:
                prompts.append(question)
                seen.add(question)

    return prompts


def _paraphrase_variants(
    base_variants: List[ResponseVariant],
    paraphraser: Paraphraser,
    edit_spans: List[int],
    similarity: EmbeddingSimilarity | None,
    min_similarity: float,
) -> tuple[List[ResponseVariant], List[ResponseVariant]]:
    """Paraphrase variants to create on/off policy pairs.

    Returns: (paired_on, paired_off) lists of ResponseVariants
    """
    paired_on: List[ResponseVariant] = []
    paired_off: List[ResponseVariant] = []

    for variant in base_variants:
        for span in edit_spans:
            # Extract think content and find non-empty lines
            content = variant.reasoning.strip()
            if content.startswith("<think>"):
                _, rest = content.split("<think>", 1)
                content = rest.split("</think>", 1)[0] if "</think>" in rest else rest

            lines = [line for line in content.strip().splitlines() if line.strip()]
            if len(lines) < span:
                continue

            # Paraphrase
            rewritten = paraphraser.rewrite(variant.reasoning, span)
            if rewritten is None:
                continue

            # Check similarity
            if similarity is not None:
                sim = similarity.similarity(
                    variant.compose(include_answer=False),
                    ResponseVariant(rewritten, variant.final_answer).compose(include_answer=False),
                )
                if sim < min_similarity:
                    continue

            paired_on.append(variant)
            paired_off.append(ResponseVariant(rewritten, variant.final_answer))

    return paired_on, paired_off


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate on/off-policy dataset")
    parser.add_argument("output", type=Path, help="Path to save dataset JSON")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Reasoning model for on-policy rollouts")
    parser.add_argument("--paraphraser", default="Qwen/Qwen2.5-7B-Instruct", help="Model to paraphrase reasoning for off-policy (use 'openrouter:MODEL' for API)")
    parser.add_argument("--paraphraser-provider", default="deepinfra", help="OpenRouter provider when using openrouter paraphraser")
    parser.add_argument("--prompts", default="persona_extract", help="Prompt source identifier")
    parser.add_argument("--traits", nargs="*", help="Traits to pull questions from (persona sources only)")
    parser.add_argument("--num-samples", type=int, default=5, help="On-policy samples per prompt")
    parser.add_argument("--max-prompts", type=int, default=10, help="Limit number of prompts during dry runs")
    parser.add_argument("--edit-spans", nargs="*", type=int, default=[1, 2, 3], help="Number of leading reasoning lines to rewrite for off-policy variants")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--similarity-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--similarity-device", default=None)
    parser.add_argument("--min-similarity", type=float, default=0.8)
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate on-policy traces and save them without paraphrasing",
    )
    parser.add_argument(
        "--paraphrase-existing",
        action="store_true",
        help="Load existing dataset and (re)generate paraphrased off-policy variants",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing dataset file (skips prompts already processed)",
    )

    args = parser.parse_args()

    if args.generate_only and args.paraphrase_existing:
        raise ValueError("--generate-only and --paraphrase-existing cannot be used together")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_metadata: Dict[str, object] = {
        "reasoning_model": args.model,
        "paraphraser": args.paraphraser,
        "prompt_source": args.prompts,
        "traits": args.traits or [],
        "num_samples": args.num_samples,
    }

    if args.paraphrase_existing:
        if not output_path.exists():
            raise RuntimeError(
                f"Cannot paraphrase without an existing dataset at {output_path}"
            )
        dataset = load_dataset(output_path)
        dataset.metadata.update(run_metadata)
    else:
        if args.resume and output_path.exists():
            dataset = load_dataset(output_path)
            print(
                f"Resuming from {output_path} with {len(dataset.examples)} existing examples"
            )
            dataset.metadata.update(run_metadata)
        else:
            if output_path.exists():
                raise RuntimeError(
                    f"Output file {output_path} already exists. Use --resume or remove the file first."
                )
            dataset = OnPolicyDataset(metadata=run_metadata)

    existing_prompts: Set[str] = set()
    for ex in dataset.examples:
        stored_question = ex.metadata.get("question") if ex.metadata else None
        existing_prompts.add(stored_question or ex.prompt)

    if args.paraphrase_existing:
        # We only paraphrase entries lacking off-policy variants
        examples_to_update = [ex for ex in dataset.examples if not ex.off_policy]
        if not examples_to_update:
            print("All examples already have off-policy variants; nothing to paraphrase.")
            return
        paraphraser = Paraphraser(
            args.paraphraser,
            provider=args.paraphraser_provider,
            requests_per_minute=100,
            device_preference=get_default_device(),
        )
        similarity = None
        if args.similarity_model.lower() != "none":
            sim_device = args.similarity_device or get_default_device()
            similarity = EmbeddingSimilarity(args.similarity_model, device=sim_device)

        updated = 0
        for example in tqdm(examples_to_update, desc="Paraphrasing"):
            base_variants = example.on_policy or []
            paired_on, paired_off = _paraphrase_variants(
                base_variants, paraphraser, args.edit_spans, similarity, args.min_similarity
            )

            if paired_on:
                example.on_policy = paired_on
                example.off_policy = paired_off
                updated += 1
                save_dataset(dataset, output_path)

        if updated == 0:
            print("No paraphrases were added. Check paraphraser configuration or similarity threshold.")
        else:
            print(
                f"Paraphrased {updated} examples. Dataset now has {len(dataset.examples)} examples."
            )

        if len(dataset.examples) == 0:
            raise RuntimeError(
                "No examples present after paraphrasing – check dataset integrity."
            )
        return

    default_traits = ["evil", "sycophantic", "hallucinating"] if not args.traits else args.traits
    prompts = _load_prompts(args.prompts, traits=default_traits)
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]

    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    if existing_prompts:
        skipped = sum(1 for prompt in prompts if prompt in existing_prompts)
        if skipped:
            print(f"Skipping {skipped} prompts already present in the dataset")

    remaining_prompts = [prompt for prompt in prompts if prompt not in existing_prompts]
    if not remaining_prompts:
        print("No new prompts to process; dataset is up to date.")
        return

    total_existing = len(dataset.examples)
    total_final = total_existing + len(remaining_prompts)

    device = get_default_device()
    dtype = torch.float16 if device != "cpu" else torch.float32
    print(f"Loading model on {device} with {dtype}")
    reasoning_model, reasoning_tokenizer = load_model_and_tokenizer(
        args.model, device_map=device, dtype=dtype
    )

    paraphraser = None
    similarity = None
    if not args.generate_only:
        paraphraser = Paraphraser(
            args.paraphraser,
            provider=args.paraphraser_provider,
            requests_per_minute=100,
            device_preference=get_default_device(),
        )
        if args.similarity_model.lower() != "none":
            sim_device = args.similarity_device or get_default_device()
            similarity = EmbeddingSimilarity(args.similarity_model, device=sim_device)

    gen_settings = GenerationSettings(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=8096,
    )

    added_examples = 0
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
            question,  # Just the raw question
            settings=gen_settings,
            num_samples=args.num_samples,
        )
        print(f"Generated {len(on_variants)} variants")

        # Paraphrase to create on/off policy pairs
        if args.generate_only:
            paired_on = on_variants
            paired_off = []
        else:
            paired_on, paired_off = _paraphrase_variants(
                on_variants, paraphraser, args.edit_spans, similarity, args.min_similarity
            )

        if not paired_on:
            print("Skipping prompt (no successful paraphrases)")
            continue

        example_id = f"example-{current_index - 1:04d}"
        example = ReasoningExample(
            example_id=example_id,
            prompt=question,
            on_policy=paired_on,
            off_policy=paired_off,
        )
        dataset.add(example)
        added_examples += 1

        save_dataset(dataset, output_path)
        print(f"Saved progress: {len(dataset.examples)} examples written to {output_path}")

    if added_examples == 0:
        print("No new examples were generated (all prompts may have failed similarity filtering).")
    else:
        print(
            f"Completed {added_examples} new prompts; dataset now has {len(dataset.examples)} examples."
        )

    if len(dataset.examples) == 0:
        raise RuntimeError(
            "No examples generated – check similarity threshold, model outputs, or prompt source."
        )


if __name__ == "__main__":
    main()
