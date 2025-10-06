#!/usr/bin/env python
"""Paraphrase each unique sentence independently (separate API calls)."""
import json
import argparse
from pathlib import Path
from typing import Set
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

def is_paraphraseable(sentence: str) -> bool:
    """Check if a sentence should be paraphrased.

    Returns False for:
    - Pure punctuation (---, $$, etc.)
    - Bold-only labels (**X**)
    """
    # Pure punctuation/symbols
    if re.match(r'^[^a-zA-Z0-9]+$', sentence):
        return False

    # Bold-only labels
    if sentence.startswith('**') and sentence.endswith('**') and sentence.count('**') == 2:
        # Check if content is very short (likely just a label)
        content = sentence.strip('*').strip()
        if len(content.split()) <= 3:  # "Pros:", "Tips:", "Unique Traits:"
            return False

    return True

def extract_sentences(raw_completion: str):
    """Extract sentences from raw completion (filter empty lines and tags)."""
    sentences = []
    for line in raw_completion.split('\n'):
        stripped = line.strip()
        # Skip empty lines, think tags, and chat template tokens
        if not stripped or stripped in ['<think>', '</think>', '<|im_start|>', '<|im_end|>']:
            continue
        sentences.append(stripped)
    return sentences

def paraphrase_sentence(client, sentence: str, model: str, system_prompt: str):
    """Paraphrase a single sentence (independent API call)."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sentence}
            ],
            temperature=0.7,
            max_tokens=4096
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default='data/on_policy.json')
    parser.add_argument('--output', type=Path, default='data/sentence_paraphrases.json')
    parser.add_argument('--models', nargs='+', default=[
        'openai/gpt-5-mini',
        'deepseek/deepseek-chat',
        'anthropic/claude-3.5-haiku'
    ])
    parser.add_argument('--api-key', help='OpenRouter API key (or set OPENROUTER_API_KEY)')
    parser.add_argument('--max-workers', type=int, default=30, help='Number of parallel API requests')
    parser.add_argument('--checkpoint-every', type=int, default=100, help='Save checkpoint every N completions')
    args = parser.parse_args()

    api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("Must provide --api-key or set OPENROUTER_API_KEY")

    # OpenRouter client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    system_prompt = (
        "Paraphrase the following sentence while preserving its exact meaning. "
        "Change the wording and structure, but keep the semantic content identical. "
        "Output only the paraphrased sentence, nothing else."
    )

    # Load on-policy data
    with open(args.input) as f:
        dataset = json.load(f)

    # Collect unique non-empty sentences
    unique_sentences: Set[str] = set()
    for example in dataset:
        for rollout in example['rollouts']:
            for sentence in extract_sentences(rollout):
                unique_sentences.add(sentence)

    # Filter to only paraphraseable sentences
    paraphraseable = [s for s in unique_sentences if is_paraphraseable(s)]
    non_paraphraseable = [s for s in unique_sentences if not is_paraphraseable(s)]

    print(f"Found {len(unique_sentences)} unique sentences")
    print(f"  Paraphraseable: {len(paraphraseable)}")
    print(f"  Non-paraphraseable (will be kept as-is): {len(non_paraphraseable)}")
    print(f"Paraphrasing with {len(args.models)} models using {args.max_workers} parallel workers...")

    # Create paraphrasing tasks (only for paraphraseable sentences)
    tasks = []
    for sentence in paraphraseable:
        for model in args.models:
            tasks.append((sentence, model))

    print(f"Total tasks: {len(tasks)}")

    # Paraphrase database: sentence -> list of paraphrases (only for paraphraseable)
    paraphrase_db = {sentence: [] for sentence in paraphraseable}

    # Parallel paraphrasing
    completed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(paraphrase_sentence, client, sentence, model, system_prompt): (sentence, model)
            for sentence, model in tasks
        }

        for future in as_completed(future_to_task):
            sentence, model = future_to_task[future]
            try:
                paraphrased = future.result()
                if paraphrased:
                    paraphrase_db[sentence].append({
                        'text': paraphrased,
                        'model': model
                    })
            except Exception as e:
                print(f"  Error paraphrasing with {model}: {e}")

            completed += 1
            if completed % 50 == 0:
                print(f"  {completed}/{len(tasks)} tasks completed...")

            # Checkpoint saving
            if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
                checkpoint_path = args.output.parent / f'{args.output.stem}_checkpoint.json'
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, 'w') as f:
                    json.dump(paraphrase_db, f, indent=2)
                print(f"  Checkpoint saved: {checkpoint_path}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(paraphrase_db, f, indent=2)

    # Stats
    total_paraphrases = sum(len(v) for v in paraphrase_db.values())
    avg_per_sentence = total_paraphrases / len(paraphrase_db) if paraphrase_db else 0
    print(f"\nSaved paraphrases for {len(paraphrase_db)} sentences")
    print(f"Total paraphrases: {total_paraphrases} (avg {avg_per_sentence:.1f} per sentence)")
    print(f"Output: {args.output}")

if __name__ == '__main__':
    main()
