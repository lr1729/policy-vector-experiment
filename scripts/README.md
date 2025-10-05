# Scripts Guide

## Current Workflow (What You Actually Need)

```bash
# Step 0 (optional): regenerate or augment the dataset
python scripts/generate_dataset.py data/on_policy_persona.json \
  --generate-base --edit-spans 1 \
  --model Qwen/Qwen3-4B --paraphraser openrouter:gpt-oss-20b

# Step 0b (optional): add multi-model paraphrases on top of a base file
python scripts/generate_dataset.py data/on_policy_persona_augmented.json \
  --augment --input data/on_policy_persona.json \
  --rollouts-per-prompt 2 --max-workers 10

# Step 1: Extract vectors from the augmented dataset (already produced)
python scripts/extract_vector.py \
  data/on_policy_persona_augmented.json \
  artifacts/qwen3_augmented_full.pt \
  --model Qwen/Qwen3-4B --reduction mean

# Step 2: Evaluate vector performance (check results)
python scripts/evaluate_vector.py \
  data/on_policy_persona_augmented.json \
  artifacts/qwen3_augmented_full.pt \
  --model Qwen/Qwen3-4B --top 10

# Step 3 (optional): Cache activations for faster iteration
python scripts/cache_activations.py \
  data/on_policy_persona_augmented.json \
  artifacts/acts_full.pkl \
  --include-answer --layers 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34
```

---

## Script Details

### ✅ **generate_dataset.py** – DATASET PIPELINE
**Purpose:** Create the base on-policy dataset and/or augment it with multi-model paraphrases.

**Key capabilities:**
- `--generate-base`: sample on-policy traces from a reasoning model and (optionally) paraphrase each rollout with a single paraphraser to produce off-policy pairs.
- `--augment`: add additional paraphrases from several models/scopes on top of an existing dataset (same logic as the old augmentation script).
- Maintains metadata used by the analysis pipeline (`question`, `base_on_variants`, paraphrase provenance, similarity scores, etc.).

**Examples:**
```bash
# Reproduce the original on-policy dataset with span-1 rewrites
python scripts/generate_dataset.py data/on_policy_persona.json \
  --generate-base --edit-spans 1 \
  --model Qwen/Qwen3-4B \
  --paraphraser openrouter:gpt-oss-20b \
  --prompt-source persona_extract --traits evil sycophantic hallucinating \
  --num-samples 5

# Add the three-model augmentation on top of an existing base file
python scripts/generate_dataset.py data/on_policy_persona_augmented.json \
  --augment --input data/on_policy_persona.json \
  --rollouts-per-prompt 2 --max-workers 10

# End-to-end: generate base + augment in one go (base file saved alongside output)
python scripts/generate_dataset.py data/on_policy_persona_augmented.json \
  --generate-base --augment --edit-spans 1 \
  --model Qwen/Qwen3-4B --paraphraser openrouter:gpt-oss-20b
```

### ✅ **extract_vector.py** – ESSENTIAL
**Purpose:** Extract the mean-difference steering vector from a dataset.

```bash
python scripts/extract_vector.py data/my_dataset.json artifacts/my_vector.pt \
  --model Qwen/Qwen3-4B --reduction mean --exclude-answer --clip-to-span
```

### ✅ **evaluate_vector.py** – ESSENTIAL
**Purpose:** Compute Cohen's d, accuracy and ranking metrics for a vector.

```bash
python scripts/evaluate_vector.py \
  data/on_policy_persona_augmented.json \
  artifacts/qwen3_augmented_full.pt \
  --model Qwen/Qwen3-4B --top 10
```

### ⚠️ **cache_activations.py** – OPTIONAL
**Purpose:** Pre-compute activations for faster notebook iteration.

```bash
python scripts/cache_activations.py \
  data/on_policy_persona_augmented.json \
  artifacts/acts_full.pkl \
  --include-answer --layers 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34
```

---

## Recommended Usage

- ✅ `generate_dataset.py` – use for any future dataset regeneration or augmentation.
- ✅ `extract_vector.py` – always run after you have a dataset.
- ✅ `evaluate_vector.py` – sanity check and reporting.
- ⚠️ `cache_activations.py` – run when repeated probe analysis is needed.

---

## Natural Next Steps After Dataset Work

1. `python scripts/extract_vector.py ...` to refresh vectors.
2. `python scripts/evaluate_vector.py ...` to validate the new vector.
3. (Optional) Regenerate cached activations if layer coverage changes.
