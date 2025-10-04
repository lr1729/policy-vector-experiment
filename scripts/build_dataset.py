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


def _first_nonempty_line_indices(reasoning: str, count: int) -> List[int]:
    """Get indices of first N non-empty content lines from reasoning.

    Extracts content between <think> tags (if present), then returns indices
    of the first N non-empty lines within that extracted content.

    Returns: List of indices where each index points to a non-empty line
             in the extracted content (after removing <think> tags).
    """
    # Extract content between <think> tags if present
    content = reasoning
    if content.strip().startswith("<think>"):
        # Handle both complete and incomplete generations
        try:
            _, rest = content.split("<think>", 1)
            if "</think>" in rest:
                content, _ = rest.split("</think>", 1)
            else:
                content = rest  # Incomplete generation (no closing tag)
            content = content.strip()
        except ValueError:
            pass  # Use as-is if malformed

    # Find indices of non-empty lines in the extracted content
    # These indices will match what the paraphraser sees after extraction
    lines = content.splitlines()
    non_empty_indices = []

    for idx, line in enumerate(lines):
        if line.strip():  # Non-empty line
            non_empty_indices.append(idx)
            if len(non_empty_indices) == count:
                break

    return non_empty_indices


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

    def _example_has_off_policy(example: ReasoningExample) -> bool:
        return bool(example.off_policy)

    if args.paraphrase_existing:
        # We only paraphrase entries lacking off-policy variants
        examples_to_update = [
            ex for ex in dataset.examples if not _example_has_off_policy(ex)
        ]
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
        for idx, example in enumerate(tqdm(examples_to_update, desc="Paraphrasing")):
            base_variants_raw = example.metadata.get("base_on_variants")
            if base_variants_raw:
                base_variants = [ResponseVariant.from_dict(v) for v in base_variants_raw]
            else:
                base_variants = [ResponseVariant.from_dict(v.to_dict()) for v in example.on_policy]

            paired_on: List[ResponseVariant] = []
            paired_off: List[ResponseVariant] = []

            for variant in base_variants:
                for span in args.edit_spans:
                    targets = _first_nonempty_line_indices(variant.reasoning, span)
                    if len(targets) < span:
                        continue

                    rewritten = paraphraser.rewrite(variant.reasoning, len(targets))
                    if rewritten is None:
                        continue

                    line_numbers = list(range(1, len(targets) + 1))
                    off_variant = ResponseVariant(
                        reasoning=rewritten,
                        final_answer=variant.final_answer,
                        metadata={
                            **variant.metadata,
                            "edit_type": "paraphrase",
                            "edit_span": span,
                            "edited_content_indices": targets,
                            "edited_line_numbers": line_numbers,
                        },
                    )

                    base_on_metadata = {
                        **variant.metadata,
                        "paired_edit_span": span,
                        "paired_content_indices": targets,
                    }

                    sim_value = None
                    if similarity is not None:
                        sim_value = float(
                            similarity.similarity(
                                variant.compose(include_answer=False),
                                off_variant.compose(include_answer=False),
                            )
                        )
                        if sim_value < args.min_similarity:
                            continue
                        off_variant.metadata["similarity"] = sim_value
                        base_on_metadata["paired_similarity"] = sim_value

                    paired_on.append(
                        ResponseVariant(
                            reasoning=variant.reasoning,
                            final_answer=variant.final_answer,
                            metadata={
                                **base_on_metadata,
                                "paired_line_numbers": line_numbers,
                            },
                        )
                    )
                    paired_off.append(off_variant)

            if not paired_on:
                print(
                    f"Example {example.example_id} produced no paraphrases passing filters; leaving as-is"
                )
                continue

            example.on_policy = paired_on
            example.off_policy = paired_off
            example.metadata["base_on_variants"] = [v.to_dict() for v in base_variants]
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

        base_variants = [
            ResponseVariant(
                reasoning=variant.reasoning,
                final_answer=variant.final_answer,
                metadata=variant.metadata.copy(),
            )
            for variant in on_variants
        ]

        paired_on: List[ResponseVariant] = []
        paired_off: List[ResponseVariant] = []

        if not args.generate_only:
            for variant in on_variants:
                for span in args.edit_spans:
                    targets = _first_nonempty_line_indices(variant.reasoning, span)
                    if len(targets) < span:
                        continue

                    rewritten_reasoning = paraphraser.rewrite(variant.reasoning, len(targets))
                    if rewritten_reasoning is None:
                        continue

                    line_numbers = list(range(1, len(targets) + 1))
                    off_variant = ResponseVariant(
                        reasoning=rewritten_reasoning,
                        final_answer=variant.final_answer,
                        metadata={
                            **variant.metadata,
                            "edit_type": "paraphrase",
                            "edit_span": span,
                            "edited_content_indices": targets,
                            "edited_line_numbers": line_numbers,
                        },
                    )

                    base_on_metadata = {
                        **variant.metadata,
                        "paired_edit_span": span,
                        "paired_content_indices": targets,
                    }
                    sim_value = None
                    if similarity is not None:
                        sim_value = float(
                            similarity.similarity(
                                variant.compose(include_answer=False),
                                off_variant.compose(include_answer=False),
                            )
                        )
                        if sim_value < args.min_similarity:
                            continue
                        off_variant.metadata["similarity"] = sim_value
                        base_on_metadata["paired_similarity"] = sim_value

                    on_clone = ResponseVariant(
                        reasoning=variant.reasoning,
                        final_answer=variant.final_answer,
                        metadata={
                            **base_on_metadata,
                            "paired_line_numbers": line_numbers,
                        },
                    )
                    paired_on.append(on_clone)
                    paired_off.append(off_variant)

        if args.generate_only:
            paired_on = base_variants

        if not paired_on:
            print("Skipping prompt (no successful paraphrases)")
            continue

        example_id = f"example-{current_index - 1:04d}"
        example = ReasoningExample(
            example_id=example_id,
            prompt=question,  # Store the original question
            on_policy=paired_on,
            off_policy=paired_off,
            metadata={
                "question": question,
                "base_on_variants": [v.to_dict() for v in base_variants],
            },
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
