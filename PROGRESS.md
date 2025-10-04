# Progress Log

- **2025-10-4**
  - Generated 60-prompt dataset with 5 Qwen/Qwen3-4B rollouts per prompt.
  - Applied OpenRouter paraphraser (first reasoning line) and similarity filter
    (Qwen3 Embedding, threshold 0.75).
  - Extracted mean-difference vector (layers 18–34, response-token averages) and
    stored at `artifacts/qwen3_onpolicy_mean.pt`.
  - Added `evaluate_vector.py` to report projection metrics (Cohen's d, accuracy).