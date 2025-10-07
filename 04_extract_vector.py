#!/usr/bin/env python
"""Extract mean-difference vector with positional clipping at actual sentence boundaries.
Generates clips at every sentence boundary (not paragraph breaks) to maximize training samples.
Uses matched_lines windowing: from response start to clip position.

Note: Paraphrasing operates at paragraph level, but extraction clips at sentence level.
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

def _token_len(tokenizer, text: str) -> int:
    return tokenizer(text, return_tensors="pt")['input_ids'].shape[1]

def _split_lines(text: str) -> List[str]:
    return text.split('\n')

def extract_sentences(text: str) -> List[str]:
    """Extract actual sentences (split on sentence boundaries: . ! ?).
    This is different from paraphrasing which operates on paragraph/line boundaries.
    """
    import re

    # Remove special tokens
    text = text.replace('<think>', '').replace('</think>', '')
    text = text.replace('<|im_start|>', '').replace('<|im_end|>', '')
    text = text.strip()

    # Split on sentence boundaries (. ! ? followed by space/newline)
    # Use a simple split that keeps the sentence together with its punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Clean up and filter
    result = []
    for sent in sentences:
        stripped = sent.strip()
        # Skip empty or very short fragments
        if stripped and len(stripped) > 5:
            result.append(stripped)

    return result

def get_hidden_states(model, tokenizer, full_text: str, output_hidden_states: bool = True):
    with torch.no_grad():
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, output_hidden_states=output_hidden_states, use_cache=False)
    return outputs.hidden_states, inputs['input_ids'].shape[1]

def mean_activation_over_window(hidden_states, start_tok: int, end_tok: int, layer_idx: int):
    """hidden_states: tuple(len_layers+1)[1, seq, d]; returns a 1D mean over a non-empty window."""
    h = hidden_states[layer_idx + 1][0]   # [seq, d]
    seq_len = h.shape[0]
    if seq_len == 0:
        # extremely defensive: return zeros if something is truly wrong
        return torch.zeros(h.shape[1]).cpu()

    # clamp to [0, seq_len]; ensure at least one token in window
    start_tok = max(0, start_tok)
    end_tok   = min(seq_len, end_tok)

    if start_tok >= seq_len:
        # boundary case: no tokens after start; back off to the last token
        start_tok = seq_len - 1
        end_tok   = seq_len
    elif end_tok <= start_tok:
        # enlarge to include exactly one token
        end_tok = start_tok + 1

    window = h[start_tok:end_tok, :]
    return window.mean(dim=0).cpu()


def build_windows_for_variant(tokenizer,
                              prompt_with_template: str,
                              on_policy_text: str,
                              off_policy_text: str) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """Return absolute token windows (start,end) for on and off texts.
       Uses matched_lines: average from response start to end of clipped text.
    """
    # Token counts
    prompt_tok_len = _token_len(tokenizer, prompt_with_template)
    on_clip_tok_len = _token_len(tokenizer, prompt_with_template + on_policy_text)
    off_clip_tok_len = _token_len(tokenizer, prompt_with_template + off_policy_text)

    # From response start to end of clip
    on_start = prompt_tok_len
    on_end = on_clip_tok_len
    off_start = prompt_tok_len
    off_end = off_clip_tok_len

    return (on_start, on_end), (off_start, off_end)

def cohens_d(list1, list2):
    diff = np.mean(list1) - np.mean(list2)
    pooled_std = np.sqrt((np.var(list1) + np.var(list2)) / 2 + 1e-8)
    return diff / (pooled_std + 1e-8)

def eval_metrics(on_proj, off_proj):
    # Balanced accuracy with simple midpoint threshold
    thr = (np.mean(on_proj) + np.mean(off_proj)) / 2.0
    on_acc = np.mean([p > thr for p in on_proj])
    off_acc = np.mean([p <= thr for p in off_proj])
    bacc = float((on_acc + off_acc) / 2.0)
    y_true = np.array([1]*len(on_proj) + [0]*len(off_proj))
    y_score = np.array(on_proj + off_proj)
    auc = None
    if roc_auc_score is not None and len(set(y_true)) == 2:
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc = None
    return thr, bacc, auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default='data/off_policy.json')
    ap.add_argument('--output', type=Path, default='artifacts/vector.json')
    ap.add_argument('--model', default='Qwen/Qwen3-4B')
    ap.add_argument('--max-sentences', type=int, default=None,
                   help='Limit positional windowing to first N sentences (default: use all)')
    ap.add_argument('--layers', nargs='+', type=int, default=None)
    args = ap.parse_args()

    # Load model
    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Layers
    num_layers = model.config.num_hidden_layers
    if args.layers is None:
        args.layers = list(range(num_layers // 2, num_layers))
    print(f"Model has {num_layers} layers -> using layers {args.layers[0]}..{args.layers[-1]}")

    # Load dataset
    with open(args.input) as f:
        dataset = json.load(f)

    # Storage
    acts = {
        'on': {l: [] for l in args.layers},
        'off': {l: [] for l in args.layers},
    }

    print("Collecting activations with matched_lines windowing")
    print("Clipping at every sentence boundary...")

    total_clips = 0
    for i, example in enumerate(dataset):
        prompt_with_template = example['prompt_with_template']
        on_rollouts = example['on_policy']

        for variant in example['off_policy']:
            src = int(variant['source_rollout_idx'])
            on_full = on_rollouts[src]
            off_full = variant.get('text', on_full)  # New format uses 'text'

            # Extract sentences for clipping
            on_sentences = extract_sentences(on_full)
            off_sentences = extract_sentences(off_full)
            max_sentences = min(len(on_sentences), len(off_sentences))

            # Optionally limit to first N sentences
            if args.max_sentences is not None:
                max_sentences = min(max_sentences, args.max_sentences)

            # Clip at every sentence boundary
            on_has_think = '<think>' in on_full
            off_has_think = '<think>' in off_full

            for clip_pos in range(1, max_sentences + 1):
                # Build clips from first N sentences
                # Join them with spaces (since sentence extractor already stripped formatting)
                on_clip = ' '.join(on_sentences[:clip_pos])
                off_clip = ' '.join(off_sentences[:clip_pos])

                if on_has_think:
                    on_clip = '<think>\n' + on_clip
                if off_has_think:
                    off_clip = '<think>\n' + off_clip

                # Compute windows
                (on_start, on_end), (off_start, off_end) = build_windows_for_variant(
                    tokenizer, prompt_with_template, on_clip, off_clip
                )

                # Full texts with prompt
                on_full_text = prompt_with_template + on_clip
                off_full_text = prompt_with_template + off_clip

                # Hidden states
                on_hs, _ = get_hidden_states(model, tokenizer, on_full_text)
                off_hs, _ = get_hidden_states(model, tokenizer, off_full_text)

                # Collect layer means
                for l in args.layers:
                    acts['on'][l].append(mean_activation_over_window(on_hs, on_start, on_end, l))
                    acts['off'][l].append(mean_activation_over_window(off_hs, off_start, off_end, l))

                total_clips += 1

    print(f"Generated {total_clips} total clips from {sum(len(ex['off_policy']) for ex in dataset)} variants")

    # Compute vectors from all data
    vector = {}
    for l in args.layers:
        on_mean = torch.stack(acts['on'][l]).mean(dim=0)
        off_mean = torch.stack(acts['off'][l]).mean(dim=0)
        vector[l] = on_mean - off_mean

    # Evaluate all layers
    print("\nEvaluation (projection onto unit vector):")
    header = "Layer | d (Cohen) | Bacc |  AUC "
    print(header)
    print("-"*len(header))

    results = []

    for l in args.layers:
        v = vector[l] / (torch.norm(vector[l]) + 1e-8)
        on_proj = [float(v @ a) for a in acts['on'][l]]
        off_proj = [float(v @ a) for a in acts['off'][l]]
        d = cohens_d(on_proj, off_proj)
        thr, bacc, auc = eval_metrics(on_proj, off_proj)
        results.append((l, d, bacc, auc, thr))
        print(f"{l:5d} | {d:9.3f} | {bacc:4.2f} | {auc if auc is not None else float('nan'):5.3f}")

    # Save all layer vectors
    total_samples = len(acts['on'][args.layers[0]])
    out = {
        'vectors': {
            str(l): vector[l].cpu().numpy().tolist()
            for l in args.layers
        },
        'total_samples': total_samples
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f)
    print(f"\nSaved {len(args.layers)} layer vectors to {args.output}")
    print(f"Total samples (per policy type): {total_samples}")

if __name__ == '__main__':
    main()
