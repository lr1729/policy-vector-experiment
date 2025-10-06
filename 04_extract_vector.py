#!/usr/bin/env python
"""Extract mean-difference vector and evaluate."""
import json
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def get_activations(model, tokenizer, prompt_with_template, raw_completion, layers):
    """Get activations for response tokens.

    Args:
        prompt_with_template: Full prompt including chat template tokens
        raw_completion: Raw completion from model (with all special tokens)
        layers: List of layer indices to extract

    Returns:
        Dict mapping layer_idx -> mean activation tensor
    """
    # Concatenate prompt and completion (exact same format as generation)
    full_text = prompt_with_template + raw_completion

    # Tokenize both
    full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_inputs = tokenizer(prompt_with_template, return_tensors="pt").to(model.device)

    prompt_len = prompt_inputs['input_ids'].shape[1]

    # Forward pass
    with torch.no_grad():
        outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)

    # Extract activations (response tokens only)
    activations = {}
    for layer_idx in layers:
        # hidden_states[0] = embeddings, hidden_states[layer_idx+1] = layer output
        hidden = outputs.hidden_states[layer_idx + 1][0]  # [seq_len, hidden_dim]
        response_hidden = hidden[prompt_len:, :]  # Response tokens only

        if response_hidden.numel() == 0:
            raise ValueError(f"No response tokens captured! Check template formatting.")

        # Mean over tokens
        activations[layer_idx] = response_hidden.mean(dim=0).cpu()

    return activations

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
    return diff / (pooled_std + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default='data/off_policy.json')
    parser.add_argument('--output', type=Path, default='artifacts/vector.pt')
    parser.add_argument('--model', default='Qwen/Qwen3-4B')
    parser.add_argument('--layers', nargs='+', type=int, default=None)
    parser.add_argument('--use-full', action='store_true', help='Use full completions instead of clipped')
    args = parser.parse_args()

    # Load model
    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Determine layers
    num_layers = model.config.num_hidden_layers
    if args.layers is None:
        # Default: second half of layers
        args.layers = list(range(num_layers // 2, num_layers))
    print(f"Model has {num_layers} layers, extracting from {len(args.layers)} layers: {args.layers[0]}-{args.layers[-1]}")

    # Load dataset
    with open(args.input) as f:
        dataset = json.load(f)

    # Collect activations
    print("Collecting activations...")
    on_acts = {l: [] for l in args.layers}
    off_acts = {l: [] for l in args.layers}

    for i, example in enumerate(dataset, 1):
        prompt_with_template = example['prompt_with_template']
        print(f"  [{i}/{len(dataset)}] Processing example...")

        # On-policy activations
        for raw_completion in example['on_policy']:
            acts = get_activations(model, tokenizer, prompt_with_template, raw_completion, args.layers)
            for l in args.layers:
                on_acts[l].append(acts[l])

        # Off-policy activations
        for variant in example['off_policy']:
            # Use full or clipped based on flag
            if args.use_full:
                completion_text = variant.get('text_full', variant.get('text'))
            else:
                completion_text = variant.get('text_clipped', variant.get('text'))
            acts = get_activations(model, tokenizer, prompt_with_template, completion_text, args.layers)
            for l in args.layers:
                off_acts[l].append(acts[l])

    # Compute mean-difference vector
    print("\nExtracting vector...")
    vector = {}
    for l in args.layers:
        on_mean = torch.stack(on_acts[l]).mean(dim=0)
        off_mean = torch.stack(off_acts[l]).mean(dim=0)
        vector[l] = on_mean - off_mean

    # Evaluate
    print("\nEvaluation:")
    print("Layer | Cohen's d | Balanced Acc | N_on | N_off")
    print("------|-----------|--------------|------|------")

    best_d = -1
    best_layer = None

    for l in args.layers:
        vec = vector[l] / (torch.norm(vector[l]) + 1e-8)  # Normalize

        # Project activations
        on_proj = [float(vec @ a) for a in on_acts[l]]
        off_proj = [float(vec @ a) for a in off_acts[l]]

        # Cohen's d
        d = cohens_d(on_proj, off_proj)

        # Accuracy
        threshold = (np.mean(on_proj) + np.mean(off_proj)) / 2
        on_acc = np.mean([p > threshold for p in on_proj])
        off_acc = np.mean([p <= threshold for p in off_proj])
        balanced_acc = (on_acc + off_acc) / 2

        print(f"{l:5} | {d:9.3f} | {balanced_acc*100:11.1f}% | {len(on_proj):4} | {len(off_proj):4}")

        if d > best_d:
            best_d = d
            best_layer = l

    print(f"\nBest layer: {best_layer} (d={best_d:.3f})")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vector, args.output)
    print(f"Saved vector to {args.output}")

if __name__ == '__main__':
    main()
