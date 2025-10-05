# Progress Log

- **2025-10-4**
  - Generated 60-prompt dataset with 5 Qwen/Qwen3-4B rollouts per prompt.
  - Applied OpenRouter paraphraser (first reasoning line) and similarity filter
    (Qwen3 Embedding, threshold 0.75).
  - Extracted mean-difference vector (layers 18–34, response-token averages) and
    stored at `artifacts/qwen3_onpolicy_mean.pt`.
  - Added `evaluate_vector.py` to report projection metrics (Cohen's d, accuracy).
- **2025-10-05**
  - Formalized interpretability goals separating detection metrics from compliance outcomes in `RESEARCH_PLAN.md`.
  - Logged current probe stats (d=2.19, layer 19) and introduced a trimmed first-sentence dataset workflow (`scripts/make_first_line_dataset.py`, `notebooks/edited_span_vector_analysis.ipynb`).
- **2025-10-06**
  - Compared mean-difference vectors extracted from (a) full reasoning + answer traces and (b) traces truncated to the first reasoning sentence only.
  - First-line-only vector (no final answer tokens) yields peak `d≈4.44` at layer 24 with >97% balanced accuracy, confirming the signal is concentrated in the paraphrased opening sentence and largely surface-form.
  - Full-trace vector remains `d≈2.19` with no detectable steering effect; documented the contrast and implications in `RESEARCH_PLAN.md`.
- **2025-10-07**
  - Rebuilt paraphrasing pipeline to generate first/mid/full edits using three paraphrasers (GPT-5 mini, DeepSeek R1, Claude Haiku) for two rollouts per prompt.
  - Added concurrent augmentation script (`scripts/augment_dataset_paraphrase.py`), producing 1,072 off-policy variants with metadata (`paraphraser`, `edit_scope`, `line_numbers`).
  - Updated vector extraction to support reasoning-only clipping (`--exclude-answer --clip-to-span`) alongside the full-rollout mode.
  - Removed legacy off-policy entries from the dataset so analyses operate solely on the new multi-model paraphrases.
