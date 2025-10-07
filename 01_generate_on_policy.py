#!/usr/bin/env python
"""Generate natural on-policy reasoning rollouts."""
import json
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

def generate_reasoning_batch(model, tokenizer, prompt, num_samples=1, max_tokens=8192):
    """Generate multiple reasoning rollouts in a single batched call."""
    messages = [{"role": "user", "content": prompt}]

    # Build full prompt with chat template (including <|im_start|>user etc)
    prompt_with_template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    inputs = tokenizer(prompt_with_template, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.eos_token_id
        )

    rollouts = []
    input_len = inputs['input_ids'].shape[1]

    for output in outputs:
        # Get completion tokens (after prompt)
        completion_ids = output[input_len:]

        # Find first EOS token (if any) and truncate there
        # This removes padding EOS but keeps all other special tokens
        eos_positions = (completion_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            # Keep everything up to and including first EOS
            first_eos = eos_positions[0].item()
            completion_ids = completion_ids[:first_eos + 1]

        # Decode with special tokens preserved (keeps <think>, </think>, etc.)
        raw_completion = tokenizer.decode(completion_ids, skip_special_tokens=False)

        # Strip <|im_end|> token from end (it's an EOS marker, not semantic content)
        raw_completion = raw_completion.rstrip('<|im_end|>').rstrip()

        rollouts.append(raw_completion)

    return prompt_with_template, rollouts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-4B')
    parser.add_argument('--prompts', type=Path, default='data/prompts.json', help='JSON file with prompts')
    parser.add_argument('--output', type=Path, default='data/on_policy.json')
    parser.add_argument('--num-rollouts', type=int, default=10, help='Rollouts per prompt')
    parser.add_argument('--max-prompts', type=int, help='Limit number of prompts')
    args = parser.parse_args()

    # Load model
    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load prompts
    with open(args.prompts) as f:
        prompts_data = json.load(f)

    if isinstance(prompts_data, list):
        prompts = prompts_data
    elif isinstance(prompts_data, dict) and 'examples' in prompts_data:
        prompts = [ex.get('prompt', ex.get('question', '')) for ex in prompts_data['examples']]
    else:
        raise ValueError("Prompts file must be list or dict with 'examples' key")

    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    # Generate rollouts
    dataset = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] {prompt[:60]}...")
        print(f"  Generating {args.num_rollouts} rollouts in batch...", end=' ')

        # Batched generation for speed
        prompt_with_template, rollouts = generate_reasoning_batch(
            model, tokenizer, prompt, num_samples=args.num_rollouts
        )

        # Extract sentences for paraphrasing (just for info/debugging)
        total_sentences = sum(len(extract_sentences(r)) for r in rollouts)
        print(f"done ({total_sentences} total sentences)")

        dataset.append({
            'prompt_with_template': prompt_with_template,
            'rollouts': rollouts
        })

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(dataset, f, indent=2)

    total_rollouts = sum(len(ex['rollouts']) for ex in dataset)
    print(f"\nSaved {len(dataset)} prompts, {total_rollouts} rollouts")
    print(f"Output: {args.output}")

if __name__ == '__main__':
    main()
