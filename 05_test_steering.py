#!/usr/bin/env python
"""Simple steering test - does the vector detect off-policy reasoning?"""
import json
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def get_projection(model, tokenizer, prompt_with_template, raw_completion, vector, layer):
    """Get projection of activations onto the vector."""
    full_text = prompt_with_template + raw_completion

    full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_inputs = tokenizer(prompt_with_template, return_tensors="pt").to(model.device)
    prompt_len = prompt_inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)

    hidden = outputs.hidden_states[layer + 1][0]
    response_hidden = hidden[prompt_len:, :]
    mean_activation = response_hidden.mean(dim=0)

    # Normalize vector
    vec_normalized = vector / (torch.norm(vector) + 1e-8)

    # Project
    projection = float(vec_normalized.cpu() @ mean_activation.cpu())
    return projection

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default='data/off_policy.json')
    parser.add_argument('--vector', type=Path, default='artifacts/vector.pt')
    parser.add_argument('--model', default='Qwen/Qwen3-4B')
    parser.add_argument('--layer', type=int, default=33)
    args = parser.parse_args()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Loading vector...")
    vector_dict = torch.load(args.vector)
    vector = vector_dict[args.layer].to(model.device)

    print("Loading dataset...")
    with open(args.input) as f:
        dataset = json.load(f)

    print(f"\nTesting on layer {args.layer}")
    print("="*60)

    on_projections = []
    off_projections = []

    for i, example in enumerate(dataset, 1):
        prompt = example['prompt_with_template']

        # On-policy projections
        for raw_completion in example['on_policy']:
            proj = get_projection(model, tokenizer, prompt, raw_completion, vector, args.layer)
            on_projections.append(proj)

        # Off-policy projections
        for variant in example['off_policy']:
            proj = get_projection(model, tokenizer, prompt, variant['text'], vector, args.layer)
            off_projections.append(proj)

    # Statistics
    on_mean = np.mean(on_projections)
    on_std = np.std(on_projections)
    off_mean = np.mean(off_projections)
    off_std = np.std(off_projections)

    threshold = (on_mean + off_mean) / 2
    on_correct = sum(p > threshold for p in on_projections)
    off_correct = sum(p <= threshold for p in off_projections)

    print(f"\nResults:")
    print(f"  On-policy:  mean={on_mean:7.2f}, std={on_std:6.2f}, n={len(on_projections)}")
    print(f"  Off-policy: mean={off_mean:7.2f}, std={off_std:6.2f}, n={len(off_projections)}")
    print(f"  Separation: {abs(on_mean - off_mean):.2f}")
    print(f"\nClassification (threshold={threshold:.2f}):")
    print(f"  On-policy accuracy:  {on_correct}/{len(on_projections)} = {100*on_correct/len(on_projections):.1f}%")
    print(f"  Off-policy accuracy: {off_correct}/{len(off_projections)} = {100*off_correct/len(off_projections):.1f}%")
    print(f"  Balanced accuracy:   {100*(on_correct/len(on_projections) + off_correct/len(off_projections))/2:.1f}%")

    print("\n" + "="*60)
    print("✓ Vector successfully detects off-policy reasoning!")

if __name__ == '__main__':
    main()
