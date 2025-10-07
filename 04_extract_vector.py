#!/usr/bin/env python
"""Extract mean-difference vector with positional clipping.
Generates clips at every sentence boundary to maximize training samples.
Uses matched_lines windowing: from response start to clip position.
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
    """Extract meaningful sentences (skip empty lines and special tokens)."""
    sentences = []
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped or stripped in ['<think>', '</think>', '<|im_start|>', '<|im_end|>']:
            continue
        sentences.append(stripped)
    return sentences

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
                              off_policy_text: str,
                              last_edited_line_idx: int) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """Return absolute token windows (start,end) for on and off texts.
       Uses matched_lines: average from response start up to end of clip position.
    """
    # Compute prefix (completion up to last edited line, inclusive)
    on_lines = _split_lines(on_policy_text)
    off_lines = _split_lines(off_policy_text)

    # Safety: bound index
    last_idx = max(-1, last_edited_line_idx)
    if last_idx >= 0:
        on_prefix = '\n'.join(on_lines[: last_idx + 1])
        off_prefix = '\n'.join(off_lines[: last_idx + 1])
    else:
        on_prefix = ''
        off_prefix = ''

    # Token counts
    prompt_tok_len = _token_len(tokenizer, prompt_with_template)
    on_prefix_tok_len = _token_len(tokenizer, prompt_with_template + on_prefix)
    off_prefix_tok_len = _token_len(tokenizer, prompt_with_template + off_prefix)

    # From response start to end of prefix
    on_start = prompt_tok_len
    off_start = prompt_tok_len
    on_end = on_prefix_tok_len
    off_end = off_prefix_tok_len

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
    ap.add_argument('--layers', nargs='+', type=int, default=None)
    ap.add_argument('--train-frac', type=float, default=0.7)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=123)
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

    # Split by prompt index
    rng = np.random.default_rng(args.seed)
    idxs = np.arange(len(dataset))
    rng.shuffle(idxs)
    n_train = int(len(idxs) * args.train_frac)
    n_val = int(len(idxs) * args.val_frac)
    train_idx = set(idxs[:n_train])
    val_idx = set(idxs[n_train:n_train+n_val])
    test_idx = set(idxs[n_train+n_val:])

    def which_split(i):
        if i in train_idx: return 'train'
        if i in val_idx: return 'val'
        return 'test'

    # Storage
    acts = {split: {'on': {l: [] for l in args.layers}, 'off': {l: [] for l in args.layers}} for split in ('train','val','test')}

    print("Collecting activations with matched_lines windowing")
    print("Clipping at every sentence boundary...")

    total_clips = 0
    for i, example in enumerate(dataset):
        split = which_split(i)
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

            # Clip at every sentence boundary
            for clip_pos in range(1, max_sentences + 1):
                # Find line index of the clip_pos-th sentence
                on_lines = _split_lines(on_full)
                off_lines = _split_lines(off_full)

                # Count sentences and track line indices
                on_sent_count = 0
                off_sent_count = 0
                on_last_line = -1
                off_last_line = -1

                for line_idx, line in enumerate(on_lines):
                    stripped = line.strip()
                    if stripped and stripped not in ['<think>', '</think>', '<|im_start|>', '<|im_end|>']:
                        on_sent_count += 1
                        if on_sent_count <= clip_pos:
                            on_last_line = line_idx
                        else:
                            break

                for line_idx, line in enumerate(off_lines):
                    stripped = line.strip()
                    if stripped and stripped not in ['<think>', '</think>', '<|im_start|>', '<|im_end|>']:
                        off_sent_count += 1
                        if off_sent_count <= clip_pos:
                            off_last_line = line_idx
                        else:
                            break

                # Build clipped text
                on_clip = '\n'.join(on_lines[:on_last_line + 1]) if on_last_line >= 0 else ''
                off_clip = '\n'.join(off_lines[:off_last_line + 1]) if off_last_line >= 0 else ''

                # Compute windows
                (on_start, on_end), (off_start, off_end) = build_windows_for_variant(
                    tokenizer, prompt_with_template, on_clip, off_clip,
                    max(on_last_line, off_last_line)
                )

                # Full texts with prompt
                on_full_text = prompt_with_template + on_clip
                off_full_text = prompt_with_template + off_clip

                # Hidden states
                on_hs, _ = get_hidden_states(model, tokenizer, on_full_text)
                off_hs, _ = get_hidden_states(model, tokenizer, off_full_text)

                # Collect layer means
                for l in args.layers:
                    acts[split]['on'][l].append(mean_activation_over_window(on_hs, on_start, on_end, l))
                    acts[split]['off'][l].append(mean_activation_over_window(off_hs, off_start, off_end, l))

                total_clips += 1

    print(f"Generated {total_clips} total clips from {sum(len(ex['off_policy']) for ex in dataset)} variants")

    # Compute vectors from TRAIN only
    vector = {}
    for l in args.layers:
        on_mean = torch.stack(acts['train']['on'][l]).mean(dim=0)
        off_mean = torch.stack(acts['train']['off'][l]).mean(dim=0)
        vector[l] = on_mean - off_mean

    # Evaluate per split
    print("\nEvaluation per split (projection onto unit vector):")
    header = "Split  | Layer | d (Cohen) | Bacc |  AUC "
    print(header)
    print("-"*len(header))

    best = {'layer': None, 'd': -1.0}
    results = []

    for split in ('train','val','test'):
        for l in args.layers:
            v = vector[l] / (torch.norm(vector[l]) + 1e-8)
            on_proj = [float(v @ a) for a in acts[split]['on'][l]]
            off_proj = [float(v @ a) for a in acts[split]['off'][l]]
            d = cohens_d(on_proj, off_proj)
            thr, bacc, auc = eval_metrics(on_proj, off_proj)
            results.append((split, l, d, bacc, auc, thr))
            print(f"{split:5} | {l:5d} | {d:9.3f} | {bacc:4.2f} | {auc if auc is not None else float('nan'):5.3f}")

            if split == 'val' and d > best['d']:
                best.update({'layer': l, 'd': d})

    print(f"\nBest layer on VAL: {best['layer']} (d={best['d']:.3f})")

    # Save only best layer vector
    out = {
        'layer': best['layer'],
        'vector': vector[best['layer']].cpu().numpy().tolist(),
        'layers_considered': args.layers,
        'total_training_samples': len(acts['train']['on'][best['layer']])
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f)
    print(f"Saved best-layer vector JSON to {args.output}")
    print(f"Training samples: {out['total_training_samples']}")

if __name__ == '__main__':
    main()
