# Analysis Notebooks

**Current Notebooks** (using new matched_lines and k_after_edit vectors):

## 01_data_audit_and_vector_extraction.ipynb
Validates dataset statistics and vector extraction quality.

**What it does:**
- Verifies data counts (40 prompts, 200 on-policy, 600 off-policy)
- Analyzes edit distribution and paraphrase model mixing
- Checks for length confounds between on-policy and off-policy
- Compares windowing methods

**Run this first** to ensure data quality.

## 02_detection_analysis.ipynb
Comprehensive detection performance analysis for both windowing methods.

**What it does:**
- Compares matched_lines vs k_after_edit vectors
- Layer-wise Cohen's d, AUC, balanced accuracy
- Stratification by edit count
- Projection distribution visualizations

**Status:** Ready to run with new vectors (200 on-policy, 600 off-policy)

## 03_interactive_probe.ipynb
Interactive tool for testing custom reasoning samples.

**Features:**
- Switch between matched_lines and k_after_edit vectors
- Edit sample reasoning and get instant projections
- Compare against training distributions
- Batch test multiple sentence variants

**Updated:** Now uses new vectors, configurable via VECTOR_MODE

## 04_steering_evaluation.ipynb
Systematic steering experiments with residual stream hooks.

**Experiments:**
- Tests both matched_lines and k_after_edit vectors
- Alpha sweep (-4 to +4)
- 4 scenarios (wallet, math, reasoning)
- 20 samples per condition (up from 5)
- Projection tracking + behavioral validation

**Status:** Ready to measure causal steering effects

---

**Archived Notebooks** (old analysis on incorrect data):

- `01_detection_analysis_old.ipynb` - Used wrong clipping (on-policy not clipped)
- `02_steering_eval_old.ipynb` - Old vector, underpowered
- `04_causal_validation_old.ipynb` - Found NO causal link (important negative result!)
- `05_high_snr_semantic_vector_old.ipynb` - Incomplete windowing experiments

These are kept for reference but should not be used for new analysis.
