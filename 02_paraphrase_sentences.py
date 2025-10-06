#!/usr/bin/env python
"""Paraphrase each unique sentence; optional semantic similarity filter."""
import json
import argparse
from pathlib import Path
from typing import Set, Optional
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

def is_paraphraseable(sentence: str) -> bool:
    if re.match(r'^[^a-zA-Z0-9]+$', sentence):
        return False
    if sentence.startswith('**') and sentence.endswith('**') and sentence.count('**') == 2:
        content = sentence.strip('*').strip()
        if len(content.split()) <= 3:
            return False
    return True

def extract_sentences(raw_completion: str):
    sentences = []
    for line in raw_completion.split('\n'):
        stripped = line.strip()
        if not stripped or stripped in ['<think>', '</think>', '<|im_start|>', '<|im_end|>']:
            continue
        sentences.append(stripped)
    return sentences

def paraphrase_sentence(client, sentence: str, model: str, system_prompt: str):
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

def embed_text(client, text: str, emb_model: Optional[str]):
    if not emb_model:
        return None
    try:
        resp = client.embeddings.create(model=emb_model, input=text)
        return resp.data[0].embedding
    except Exception as e:
        print(f"  Embedding error: {e}")
        return None

def cos_sim(a, b):
    import math
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / max(1e-8, (na*nb))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default='data/on_policy.json')
    ap.add_argument('--output', type=Path, default='data/sentence_paraphrases.json')
    ap.add_argument('--models', nargs='+', default=[
        'openai/gpt-5-mini',
        'deepseek/deepseek-chat',
        'anthropic/claude-3.5-haiku'
    ])
    ap.add_argument('--api-key', help='OpenRouter API key (or set OPENROUTER_API_KEY)')
    ap.add_argument('--embedding-model', default=None,
                   help='Optional embedding model id (e.g., openai/text-embedding-3-large)')
    ap.add_argument('--similarity-threshold', type=float, default=0.85,
                   help='Min cosine similarity to accept a paraphrase when embedding model is set.')
    ap.add_argument('--max-workers', type=int, default=30)
    ap.add_argument('--checkpoint-every', type=int, default=100)
    args = ap.parse_args()

    api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("Must provide --api-key or set OPENROUTER_API_KEY")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    system_prompt = (
        "Paraphrase the following sentence while preserving its exact meaning. "
        "Change wording and structure, but keep the semantic content identical. "
        "Output only the paraphrased sentence, nothing else."
    )

    with open(args.input) as f:
        dataset = json.load(f)

    unique_sentences: Set[str] = set()
    for example in dataset:
        for rollout in example['rollouts']:
            for sentence in extract_sentences(rollout):
                unique_sentences.add(sentence)

    paraphraseable = [s for s in unique_sentences if is_paraphraseable(s)]
    print(f"Found {len(unique_sentences)} unique sentences")
    print(f"  Paraphraseable: {len(paraphraseable)}")

    tasks = []
    for sentence in paraphraseable:
        for model in args.models:
            tasks.append((sentence, model))

    paraphrase_db = {s: [] for s in paraphraseable}
    completed = 0

    # Pre-embed originals if embedding model set
    emb_model = args.embedding_model
    orig_emb = {}
    if emb_model:
        print("Embedding originals for semantic filtering...")
        for s in paraphraseable:
            e = embed_text(client, s, emb_model)
            if e is not None:
                orig_emb[s] = e

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(paraphrase_sentence, client, sentence, model, system_prompt): (sentence, model)
            for sentence, model in tasks
        }
        for future in as_completed(future_to_task):
            sentence, model = future_to_task[future]
            paraphrased = future.result()
            if paraphrased:
                if emb_model and sentence in orig_emb:
                    pe = embed_text(client, paraphrased, emb_model)
                    if pe is not None:
                        sim = cos_sim(orig_emb[sentence], pe)
                        if sim < args.similarity_threshold:
                            # reject
                            completed += 1
                            continue
                paraphrase_db[sentence].append({'text': paraphrased, 'model': model})
            completed += 1
            if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
                ckpt = args.output.parent / f'{args.output.stem}_checkpoint.json'
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                with open(ckpt, 'w') as f:
                    json.dump(paraphrase_db, f, indent=2)
                print(f"Checkpoint saved: {ckpt}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(paraphrase_db, f, indent=2)
    total_paraphrases = sum(len(v) for v in paraphrase_db.values())
    avg_per_sentence = total_paraphrases / max(1, len(paraphrase_db))
    print(f"Saved paraphrases for {len(paraphrase_db)} sentences")
    print(f"Total paraphrases: {total_paraphrases} (avg {avg_per_sentence:.1f}/sentence)")
    print(f"Output: {args.output}")

if __name__ == '__main__':
    main()
