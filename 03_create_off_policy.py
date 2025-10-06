#!/usr/bin/env python
"""Create off-policy variants by replacing sentences with paraphrases.
Adds metadata needed for matched-window and K-after-edit extraction.
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

def extract_sentences_with_positions(raw_completion: str) -> List[Tuple[int, str]]:
    """Extract sentences with their line positions.
    Returns: list of (line_idx, sentence_text)
    """
    sentences = []
    lines = raw_completion.split('\n')
    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped in ['<think>', '</think>', '<|im_start|>', '<|im_end|>']:
            continue
        sentences.append((line_idx, stripped))
    return sentences

def apply_paraphrases_and_clip(raw_completion: str,
                               sentences_with_pos: List[Tuple[int, str]],
                               positions_to_edit: List[int],
                               paraphrase_db: Dict[str, list]):
    """Apply paraphrases to specific sentence positions and clip at last edit.
    Returns:
        clipped_result (str): completion up to last edited line (inclusive).
        full_result (str): full completion with paraphrases.
        last_edited_line_idx (int)
        chosen_info (list[dict]): [{"line_idx": int, "orig": str, "paraphrase": str, "model": str}]
    """
    lines = raw_completion.split('\n')
    line_replacements = {}
    last_edited_line_idx = -1
    chosen_info = []

    for sent_idx in sorted(positions_to_edit):
        if sent_idx >= len(sentences_with_pos):
            continue
        line_idx, original_sentence = sentences_with_pos[sent_idx]

        # Pick random paraphrase
        options = paraphrase_db.get(original_sentence, [])
        if not options:
            # Shouldn't happen if caller checked
            continue
        chosen = random.choice(options)
        paraphrase = chosen.get('text', '')
        model = chosen.get('model', 'unknown')

        line_replacements[line_idx] = paraphrase
        last_edited_line_idx = max(last_edited_line_idx, line_idx)
        chosen_info.append({
            "line_idx": line_idx,
            "orig": original_sentence,
            "paraphrase": paraphrase,
            "model": model
        })

    # Build full result with replacements
    full_lines = []
    for line_idx, line in enumerate(lines):
        full_lines.append(line_replacements.get(line_idx, line))

    # Clip at last edited line (inclusive)
    if last_edited_line_idx != -1:
        clipped_lines = full_lines[: last_edited_line_idx + 1]
    else:
        clipped_lines = full_lines

    full_result = '\n'.join(full_lines)
    clipped_result = '\n'.join(clipped_lines)
    return clipped_result, full_result, last_edited_line_idx, chosen_info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--on-policy', type=Path, default='data/on_policy.json')
    ap.add_argument('--paraphrases', type=Path, default='data/sentence_paraphrases.json')
    ap.add_argument('--output', type=Path, default='data/off_policy.json')
    ap.add_argument('--variants-per-rollout', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # Load data
    with open(args.on_policy) as f:
        on_policy_data = json.load(f)
    with open(args.paraphrases) as f:
        paraphrase_db = json.load(f)

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

            sentences_with_pos = extract_sentences_with_positions(raw_completion)
            sentence_texts = [s for _, s in sentences_with_pos]
            paraphraseable_sentences = [s for s in sentence_texts if is_paraphraseable(s)]
            has_all = all(s in paraphrase_db and paraphrase_db[s] for s in paraphraseable_sentences)
            if not has_all:
                continue

            paraphraseable_positions = [
                i for i, (_, s) in enumerate(sentences_with_pos) if is_paraphraseable(s)
            ]
            num_paraphraseable = len(paraphraseable_positions)
            if num_paraphraseable == 0:
                continue

            for _ in range(args.variants_per_rollout):
                num_to_edit = random.randint(1, num_paraphraseable)
                chosen_positions = sorted(random.sample(paraphraseable_positions, num_to_edit))

                clipped, full, last_idx, chosen_info = apply_paraphrases_and_clip(
                    raw_completion, sentences_with_pos, chosen_positions, paraphrase_db
                )

                off_policy_variants.append({
                    'text_clipped': clipped,
                    'text_full': full,
                    'source_rollout_idx': rollout_idx,
                    'num_edits': num_to_edit,
                    'total_paraphraseable': num_paraphraseable,
                    'last_edited_line_idx': last_idx,
                    'edited_positions': chosen_positions,
                    'replacements': chosen_info
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

    # Stats
    edit_counts = {}
    for ex in dataset:
        for v in ex['off_policy']:
            k = v['num_edits']
            edit_counts[k] = edit_counts.get(k, 0) + 1

    print(f"Created dataset: {len(dataset)} prompts")
    print(f"  {total_on} on-policy rollouts")
    print(f"  {total_off} off-policy variants")
    print("  Edit distribution:", dict(sorted(edit_counts.items())))
    print(f"Output: {args.output}")
    
if __name__ == '__main__':
    main()
