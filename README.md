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
```
python scripts/build_dataset.py data/on_policy_persona.json \
  --paraphrase-existing \
  --paraphraser "openrouter:gpt-oss-20b" \
  --paraphraser-provider deepinfra \
  --edit-spans 1 \
  --temperature 0.6 \
  --top-p 0.95 \
  --similarity-model Qwen/Qwen3-Embedding-0.6B \
  --min-similarity 0.75
```

### 3. Extract Vector
```
python scripts/extract_vector.py data/on_policy_persona.json \
  artifacts/qwen3_onpolicy_mean.pt \
  --model Qwen/Qwen3-4B \
  --reduction mean
```

### 4. Evaluate Vector
```
PYTHONPATH=. python scripts/evaluate_vector.py \
  data/on_policy_persona.json \
  artifacts/qwen3_onpolicy_mean.pt \
  --model Qwen/Qwen3-4B \
  --top 5
```

The provided dataset and vector allow collaborators to run evaluation directly
without re-generating data.
