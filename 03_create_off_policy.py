#!/usr/bin/env python
"""Create off-policy variants by paraphrasing ALL paragraphs (100% paraphrasing).
Creates 3 variants per rollout: one using GPT paraphrases, one using Claude, one using DeepSeek.
No model mixing within a variant (consistent style per variant).

Note: "Paraphrasing" operates at paragraph/line-break level, not individual sentences.
Windowing/clipping at actual sentence level is handled in extraction (04_extract_vector.py).
"""
import json
import argparse
from pathlib import Path
import random
import re
from typing import List, Tuple, Dict, Any

def is_paraphraseable(sentence: str) -> bool:
    """Check if a sentence should be paraphrased.
    Must match the logic in 02_paraphrase_sentences.py.
    """
    if re.match(r'^[^a-zA-Z0-9]+$', sentence):
        return False
    if sentence.startswith('**') and sentence.endswith('**') and sentence.count('**') == 2:
        content = sentence.strip('*').strip()
        if len(content.split()) <= 3:
            return False
    return True

def extract_paragraphs_with_positions(raw_completion: str) -> List[Tuple[int, str]]:
    """Extract paragraphs (line-break delimited) with their line positions.
    Returns: list of (line_idx, paragraph_text)
    """
    sentences = []
    lines = raw_completion.split('\n')
    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped in ['<think>', '</think>', '<|im_start|>', '<|im_end|>']:
            continue
        sentences.append((line_idx, stripped))
    return sentences

def normalize_text(text: str) -> str:
    """Remove special control tokens and extra whitespace."""
    for token in ('<|im_start|>', '<|im_end|>', '<think>', '</think>'):
        text = text.replace(token, '')
    return text.strip()

def apply_paraphrases_single_model(raw_completion: str,
                                    paragraphs_with_pos: List[Tuple[int, str]],
                                    positions_to_edit: List[int],
                                    paraphrase_db: Dict[str, list],
                                    paraphrase_lookup: Dict[str, list],
                                    target_model: str):
    """Apply paraphrases from a specific model to ALL specified paragraph positions.
    Uses only paraphrases from target_model (e.g., 'openai/gpt-5-mini').

    Returns:
        paraphrased_text (str): full completion with all paragraphs paraphrased
        chosen_info (list[dict]): [{"line_idx": int, "orig": str, "paraphrase": str, "model": str}]
    """
    lines = raw_completion.split('\n')
    line_replacements = {}
    chosen_info = []

    for para_idx in positions_to_edit:
        if para_idx >= len(paragraphs_with_pos):
            continue
        line_idx, original_paragraph = paragraphs_with_pos[para_idx]

        # Pick paraphrase from target model only
        normalized = normalize_text(original_paragraph)
        options = paraphrase_lookup.get(normalized, [])
        if not options:
            options = paraphrase_db.get(original_paragraph, [])
        if not options:
            continue

        # Filter to target model
        model_options = [opt for opt in options if opt.get('model') == target_model]
        if not model_options:
            continue

        chosen = model_options[0]  # Should only be one per model
        paraphrase = chosen.get('text', '')
        model = chosen.get('model', 'unknown')

        line_replacements[line_idx] = paraphrase
        chosen_info.append({
            "line_idx": line_idx,
            "orig": original_paragraph,
            "paraphrase": paraphrase,
            "model": model
        })

    # Build full result with replacements
    result_lines = []
    for line_idx, line in enumerate(lines):
        result_lines.append(line_replacements.get(line_idx, line))

    paraphrased_text = '\n'.join(result_lines)
    return paraphrased_text, chosen_info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--on-policy', type=Path, default='data/on_policy.json')
    ap.add_argument('--paraphrases', type=Path, default='data/sentence_paraphrases.json')
    ap.add_argument('--output', type=Path, default='data/off_policy.json')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # Load data
    with open(args.on_policy) as f:
        on_policy_data = json.load(f)
    with open(args.paraphrases) as f:
        paraphrase_db = json.load(f)

    paraphrase_lookup: Dict[str, list] = {}
    for original, options in paraphrase_db.items():
        normalized = normalize_text(original)
        if not normalized:
            continue
        bucket = paraphrase_lookup.setdefault(normalized, [])
        bucket.extend(options)

    dataset = []
    total_on = 0
    total_off = 0

    for example in on_policy_data:
        prompt_with_template = example['prompt_with_template']
        on_policy_rollouts = []
        off_policy_variants = []

        for rollout_idx, raw_completion in enumerate(example['rollouts']):
            on_policy_rollouts.append(raw_completion)
            total_on += 1

            paragraphs_with_pos = extract_paragraphs_with_positions(raw_completion)
            normalized_texts = [normalize_text(s) for _, s in paragraphs_with_pos]
            paraphraseable_paragraphs = [
                normalized_texts[i]
                for i in range(len(normalized_texts))
                if normalized_texts[i] and is_paraphraseable(normalized_texts[i])
            ]
            has_all = all(paraphrase_lookup.get(s) for s in paraphraseable_paragraphs)
            if not has_all:
                continue

            paraphraseable_positions = [
                i for i, s in enumerate(normalized_texts)
                if s and is_paraphraseable(s)
            ]
            num_paraphraseable = len(paraphraseable_positions)
            if num_paraphraseable == 0:
                continue

            # Create 3 variants: one per paraphraser model (no mixing within variant)
            paraphrasers = [
                'openai/gpt-5-mini',
                'deepseek/deepseek-chat',
                'anthropic/claude-3.5-haiku'
            ]

            for model in paraphrasers:
                paraphrased_text, chosen_info = apply_paraphrases_single_model(
                    raw_completion, paragraphs_with_pos, paraphraseable_positions,
                    paraphrase_db, paraphrase_lookup, model
                )

                off_policy_variants.append({
                    'text': paraphrased_text,
                    'source_rollout_idx': rollout_idx,
                    'paraphraser_model': model
                })
                total_off += 1

        dataset.append({
            'prompt_with_template': prompt_with_template,
            'on_policy': on_policy_rollouts,
            'off_policy': off_policy_variants
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Created dataset: {len(dataset)} prompts")
    print(f"  On-policy: {total_on} rollouts")
    print(f"  Off-policy: {total_off} variants (100% paraphrased)")
    print(f"Output: {args.output}")
    
if __name__ == '__main__':
    main()
