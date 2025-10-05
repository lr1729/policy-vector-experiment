# Policy Vector Pipeline
This includes everything needed to:

1. Generate paired on-policy/off-policy reasoning traces for Qwen/Qwen3-4B
   (with a generation-only GPU pass and local paraphrasing pass).
2. Extract mean-difference steering vectors from the dataset.
3. Evaluate the resulting vector via projection statistics.
4. Apply steering vectors during generation (via included `persona_vectors` module).

## Directory Layout

- `policy_vector_pipeline/` – core Python package (generation, paraphrasing,
  similarity, activation collection, steering helpers).
- `persona_vectors/` – activation steering implementation (for `SteeringRunner`).
- `scripts/` – runnable entry points:
  - `build_dataset.py`
  - `extract_vector.py`
  - `evaluate_vector.py`
  - `augment_dataset_paraphrase.py`
  - `make_first_line_dataset.py`
- `prompts/trait_data_*` – persona prompt JSONs required by the dataset builder.
- `data/on_policy_persona.json` – sample dataset generated with 60 prompts ×
  5 rollouts.
- `artifacts/qwen3_onpolicy_mean.pt` – mean-difference vector extracted from the
  dataset (layers 18–34).
- `notebooks/` – Jupyter notebooks for exploration and evaluation.
- `requirements.txt` – all dependencies

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=...  # if you plan to paraphrase
```

### 1. Generate (GPU box)
```
python scripts/build_dataset.py data/on_policy_persona.json \
  --model Qwen/Qwen3-4B \
  --paraphraser "openrouter:gpt-oss-20b" \
  --paraphraser-provider deepinfra \
  --prompts persona_extract \
  --traits evil sycophantic hallucinating \
  --num-samples 3 \
  --max-prompts 0 \
  --edit-spans 1 \
  --temperature 0.6 \
  --top-p 0.95 \
  --similarity-model Qwen/Qwen3-Embedding-0.6B \
  --min-similarity 0.75 \
  --generate-only \
  --resume
```

### 2. Paraphrase (local)
Use the multi-model augmentation script to generate first/mid/full variants for each prompt. The example below paraphrases two rollouts per prompt with OpenAI GPT‑5 mini, DeepSeek R1, and Claude 3.5 Haiku, writing the result to `data/on_policy_persona_augmented.json`:

```
PYTHONPATH=. python scripts/augment_dataset_paraphrase.py \
  --dataset data/on_policy_persona.json \
  --output data/on_policy_persona_augmented.json \
  --rollouts-per-prompt 2 \
  --models openrouter:openai/gpt-5-mini openrouter:deepseek/deepseek-r1-0528 openrouter:anthropic/claude-3.5-haiku \
  --max-workers 10
```

Each variant records metadata (`paraphraser`, `edit_scope`, `line_numbers`, `source_rollout`) for downstream analysis.

### 3. Extract Vector
*Full reasoning + final answer*
```
PYTHONPATH=. python scripts/extract_vector.py \
  data/on_policy_persona_augmented.json \
  artifacts/qwen3_augmented_full.pt \
  --model Qwen/Qwen3-4B \
  --reduction mean
```

*Clipped reasoning span (no final answer tokens)*
```
PYTHONPATH=. python scripts/extract_vector.py \
  data/on_policy_persona_augmented.json \
  artifacts/qwen3_augmented_clipped.pt \
  --model Qwen/Qwen3-4B \
  --reduction mean \
  --exclude-answer \
  --clip-to-span
```

### 4. Evaluate Vector
```
PYTHONPATH=. python scripts/evaluate_vector.py \
  data/on_policy_persona_augmented.json \
  artifacts/qwen3_augmented_full.pt \
  --model Qwen/Qwen3-4B \
  --top 5
```

The augmented dataset now contains 291 on-policy rollouts and 1,072 off-policy
variants spanning first/mid/full edits from all three paraphrasers. Use the
clipped vector when you care about the local edit window, and the full vector
when you want the detector to consider the entire reasoning trace.
