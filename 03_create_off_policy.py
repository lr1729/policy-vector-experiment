#!/usr/bin/env python
"""Create off-policy variants by replacing sentences with paraphrases."""
import json
import argparse
from pathlib import Path
import random
import re

def is_paraphraseable(sentence: str) -> bool:
    """Check if a sentence should be paraphrased.

    Must match the logic in 02_paraphrase_sentences.py.
    """
    # Pure punctuation/symbols
    if re.match(r'^[^a-zA-Z0-9]+$', sentence):
        return False

    # Bold-only labels
    if sentence.startswith('**') and sentence.endswith('**') and sentence.count('**') == 2:
        content = sentence.strip('*').strip()
        if len(content.split()) <= 3:
            return False

    return True

def extract_sentences_with_positions(raw_completion: str):
    """Extract sentences with their line positions.

    Returns:
        List of tuples: (line_idx, sentence_text)
    """
    sentences = []
    lines = raw_completion.split('\n')
    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        # Skip empty lines, think tags, and chat template tokens
        if not stripped or stripped in ['<think>', '</think>', '<|im_start|>', '<|im_end|>']:
            continue
        sentences.append((line_idx, stripped))
    return sentences

def apply_paraphrases_and_clip(raw_completion: str, sentences_with_positions: list,
                               positions_to_edit: set, paraphrase_db: dict):
    """Apply paraphrases to specific sentence positions and clip at last edit.

    Args:
        raw_completion: Original raw completion
        sentences_with_positions: List of (line_idx, sentence_text) tuples
        positions_to_edit: Set of sentence indices (0-based) to paraphrase
        paraphrase_db: Dict mapping sentence text -> list of paraphrase options

    Returns:
        Tuple of (clipped_completion, full_completion)
    """
    lines = raw_completion.split('\n')

    # Track which line indices get edited and their replacements
    line_replacements = {}
    last_edited_line_idx = -1

    for sent_idx in sorted(positions_to_edit):
        if sent_idx >= len(sentences_with_positions):
            continue
        line_idx, original_sentence = sentences_with_positions[sent_idx]

        # Pick random paraphrase
        chosen = random.choice(paraphrase_db[original_sentence])
        paraphrase = chosen['text']

        line_replacements[line_idx] = paraphrase
        last_edited_line_idx = max(last_edited_line_idx, line_idx)

    # Build full result with replacements
    full_lines = []

    for line_idx, line in enumerate(lines):
        if line_idx in line_replacements:
            full_lines.append(line_replacements[line_idx])
        else:
            full_lines.append(line)

    # Clip at last edited line
    if last_edited_line_idx != -1:
        clipped_lines = full_lines[:last_edited_line_idx + 1]
    else:
        clipped_lines = full_lines

    full_result = '\n'.join(full_lines)
    clipped_result = '\n'.join(clipped_lines)

    return clipped_result, full_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--on-policy', type=Path, default='data/on_policy.json')
    parser.add_argument('--paraphrases', type=Path, default='data/sentence_paraphrases.json')
    parser.add_argument('--output', type=Path, default='data/off_policy.json')
    parser.add_argument('--variants-per-rollout', type=int, default=3,
                        help='Number of mixed off-policy variants per rollout')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load data
    with open(args.on_policy) as f:
        on_policy_data = json.load(f)

    with open(args.paraphrases) as f:
        paraphrase_db = json.load(f)

    # Create dataset
    dataset = []
    total_on = 0
    total_off = 0

    for example in on_policy_data:
        prompt_with_template = example['prompt_with_template']
        on_policy_rollouts = []
        off_policy_variants = []

        for rollout_idx, raw_completion in enumerate(example['rollouts']):
            # Store on-policy rollout as-is
            on_policy_rollouts.append(raw_completion)
            total_on += 1

            # Extract sentences with positions
            sentences_with_pos = extract_sentences_with_positions(raw_completion)
            sentence_texts = [sent for _, sent in sentences_with_pos]

            # Check if all PARAPHRASEABLE sentences have paraphrases
            # Non-paraphraseable sentences (---, $$, etc.) are kept as-is
            paraphraseable_sentences = [s for s in sentence_texts if is_paraphraseable(s)]
            has_all_paraphrases = all(
                sentence in paraphrase_db and paraphrase_db[sentence]
                for sentence in paraphraseable_sentences
            )

            if not has_all_paraphrases:
                continue

            # Create off-policy variants with uniform distribution across edit counts
            # Only select from paraphraseable sentence positions
            paraphraseable_positions = [
                i for i, (_, sent) in enumerate(sentences_with_pos)
                if is_paraphraseable(sent)
            ]
            num_paraphraseable = len(paraphraseable_positions)

            if num_paraphraseable == 0:
                continue

            # Generate variants_per_rollout samples uniformly across [1, N] edits
            for variant_idx in range(args.variants_per_rollout):
                # Truly random uniform sampling of number of sentences to edit
                num_to_edit = random.randint(1, num_paraphraseable)

                # Randomly select which PARAPHRASEABLE sentence positions to paraphrase
                positions_to_edit = set(random.sample(paraphraseable_positions, num_to_edit))

                # Apply paraphrases and clip at last edit
                clipped, full = apply_paraphrases_and_clip(
                    raw_completion, sentences_with_pos, positions_to_edit, paraphrase_db
                )

                off_policy_variants.append({
                    'text_clipped': clipped,  # For activation extraction
                    'text_full': full,  # For reference
                    'source_rollout_idx': rollout_idx,
                    'num_edits': num_to_edit,
                    'total_paraphraseable': num_paraphraseable
                })
                total_off += 1

        dataset.append({
            'prompt_with_template': prompt_with_template,
            'on_policy': on_policy_rollouts,
            'off_policy': off_policy_variants
        })

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(dataset, f, indent=2)

    # Compute edit distribution stats
    edit_counts = {}
    for example in dataset:
        for variant in example['off_policy']:
            k = variant['num_edits']
            edit_counts[k] = edit_counts.get(k, 0) + 1

    print(f"Created dataset:")
    print(f"  {len(dataset)} prompts")
    print(f"  {total_on} on-policy variants")
    print(f"  {total_off} off-policy variants")
    print(f"  Edit distribution: {dict(sorted(edit_counts.items()))}")
    print(f"Output: {args.output}")

if __name__ == '__main__':
    main()
