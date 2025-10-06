#!/usr/bin/env python
"""Extract mean-difference vector with windowed options and holdout eval."""
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

def get_hidden_states(model, tokenizer, full_text: str, output_hidden_states: bool = True):
    with torch.no_grad():
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, output_hidden_states=output_hidden_states, use_cache=False)
    return outputs.hidden_states, inputs['input_ids'].shape[1]

def mean_activation_over_window(hidden_states, start_tok: int, end_tok: int, layer_idx: int):
    """hidden_states: tuple(len_layers+1)[batch=1, seq, d]
       layer_idx: layer number in [0..L-1]; hidden_states[layer_idx+1] is layer output
    """
    h = hidden_states[layer_idx + 1][0]   # [seq, d]
    start_tok = max(0, start_tok)
    end_tok = min(h.shape[0], end_tok)
    if end_tok <= start_tok:
        # fallback: single token at start_tok if available
        end_tok = min(h.shape[0], start_tok + 1)
    window = h[start_tok:end_tok, :]
    return window.mean(dim=0).cpu()

def build_windows_for_variant(tokenizer,
                              prompt_with_template: str,
                              on_policy_text: str,
                              off_policy_text: str,
                              last_edited_line_idx: int,
                              window_mode: str,
                              k_tokens: int) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """Return absolute token windows (start,end) for on and off full_texts.
       Strategy:
         - matched_lines: average from response start up to end of last edited line
         - k_after_edit: average over the first K tokens after the last edited line boundary
    """
    # compute prefix (completion up to last edited line, inclusive)
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
    on_full_tok_len  = _token_len(tokenizer, prompt_with_template + on_policy_text)
    off_full_tok_len = _token_len(tokenizer, prompt_with_template + off_policy_text)

    if window_mode == 'matched_lines':
        # from response start to end of prefix
        on_start = prompt_tok_len
        off_start = prompt_tok_len
        on_end = on_prefix_tok_len
        off_end = off_prefix_tok_len
    elif window_mode == 'k_after_edit':
        # first K tokens after the boundary, independent for each condition
        on_start = on_prefix_tok_len
        off_start = off_prefix_tok_len
        on_end = min(on_full_tok_len, on_start + k_tokens)
        off_end = min(off_full_tok_len, off_start + k_tokens)
    else:
        raise ValueError("Unknown window_mode")

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
    ap.add_argument('--output', type=Path, default='artifacts/vector.pt')
    ap.add_argument('--model', default='Qwen/Qwen3-4B')
    ap.add_argument('--window', choices=['matched_lines', 'k_after_edit'], default='matched_lines')
    ap.add_argument('--k-tokens', type=int, default=24)
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

    print("Collecting activations with window:", args.window, " K=", args.k_tokens)
    for i, example in enumerate(dataset):
        split = which_split(i)
        prompt_with_template = example['prompt_with_template']

        # build on-policy lookup
        on_rollouts = example['on_policy']

        # Precompute hidden states for each on-policy rollout full text (only once per layer usage)
        # We'll compute windows per off-policy variant
        for variant in example['off_policy']:
            src = int(variant['source_rollout_idx'])
            last_idx = int(variant.get('last_edited_line_idx', -1))

            # texts (full) for building windows
            on_full = on_rollouts[src]
            # pick which off text to use (clipped or full?), here we always use full and then window it
            off_full = variant.get('text_full', variant.get('text_clipped', ''))

            # windows
            (on_start, on_end), (off_start, off_end) = build_windows_for_variant(
                tokenizer, prompt_with_template, on_full, off_full, last_idx, args.window, args.k_tokens
            )

            # Full texts
            on_full_text = prompt_with_template + on_full
            off_full_text = prompt_with_template + off_full

            # Hidden states and sequence lengths
            on_hs, _ = get_hidden_states(model, tokenizer, on_full_text)
            off_hs, _ = get_hidden_states(model, tokenizer, off_full_text)

            # Collect layer means
            for l in args.layers:
                acts[split]['on'][l].append(mean_activation_over_window(on_hs, on_start, on_end, l))
                acts[split]['off'][l].append(mean_activation_over_window(off_hs, off_start, off_end, l))

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
        'window': args.window,
        'k_tokens': args.k_tokens,
        'layers_considered': args.layers
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f)
    print(f"Saved best-layer vector JSON to {args.output}")

if __name__ == '__main__':
    main()
