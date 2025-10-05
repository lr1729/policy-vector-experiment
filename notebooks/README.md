# Notebook Guide
- `edited_span_vector_analysis.ipynb` – primary workflow for re-extracting the mean-difference vector using the full dataset and a trimmed first-sentence dataset (`data/on_policy_first_line.json`, generate with `scripts/make_first_line_dataset.py`; Qwen/Qwen3-4B rollouts, paraphrased with openrouter:gpt-oss-20b; outputs under `artifacts/edited_span_runs/`).
- `statistical_validation.ipynb` – permutation tests and cross-validation for any saved vector.
- `confound_analysis.ipynb` – lexical fingerprint diagnostics; run after generating a new vector to confirm paraphraser artifacts are reduced.
- `injection_acceptance_test.ipynb` – compliance benchmarking across ethical, preference, and factual scenarios.
- `dataset_explorer.ipynb` – quick inspection utilities for `data/on_policy_persona.json` (sanity checks, sampling).
