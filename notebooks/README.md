# Notebooks Guide

## Active Notebooks (Use These)

### **1. augmented_vector_analysis.ipynb**
**Purpose:** Comprehensive probe performance analysis of multi-model paraphrased vectors

**What it does:**
- Loads FULL and CLIPPED vectors
- Analyzes Cohen's d, accuracy across all paraphrasers
- Tests cross-paraphraser generalization (key finding: std=0.062!)
- Compares to original single-paraphraser results
- Saves best layer config for steering

**When to run:** After extracting new vectors, to evaluate probe performance

**Results:**
- FULL vector: d=1.52, acc=78.5%, layer 34
- CLIPPED vector: d=1.81, acc=83.6%, layer 25
- Excellent cross-paraphraser generalization ✓

---

### **2. augmented_vector_steering_test.ipynb**
**Purpose:** Initial steering effectiveness test (n=1 sample per condition)

**What it does:**
- Tests wallet/tea/lyon scenarios
- Sweeps α ∈ [0, 1, 2, 3, 4]
- Compares to original failed steering results
- Measures compliance with injected reasoning

**When to run:** Quick check of steering effectiveness

**Results:**
- **BREAKTHROUGH:** 2/3 scenarios passed (wallet +0.33, tea +0.53)
- Lyon still fails (factual knowledge too strong)
- FULL vector outperforms CLIPPED for steering strength

**Limitations:** Only n=1 sample, no statistical tests

---

### **3. comprehensive_steering_test.ipynb** ⭐ **USE THIS**
**Purpose:** Rigorous steering test with bidirectional control and statistics

**What it does:**
- Tests BOTH positive and negative α (-4 to +4)
- n=10 samples per condition (statistical significance!)
- Better factual question (Myanmar vs Lyon)
- Paired t-tests, Cohen's d, error bars
- Tests bidirectional causality

**When to run:** For publication-quality steering results

**Estimated time:** 30-45 minutes (540 generations)

**Improvements over #2:**
- ✅ Statistical significance (n=10 vs n=1)
- ✅ Bidirectional testing (confirms causality)
- ✅ Better factual scenario
- ✅ Effect size quantification

---

### **4. dataset_explorer.ipynb**
**Purpose:** Quick exploration of dataset contents

---

## Workflow

**Probe Analysis:**
```bash
jupyter notebook notebooks/augmented_vector_analysis.ipynb
```

**Steering (Quick):**
```bash
jupyter notebook notebooks/augmented_vector_steering_test.ipynb
```

**Steering (Rigorous):**
```bash
jupyter notebook notebooks/comprehensive_steering_test.ipynb
```

---

## Key Findings

- Multi-model paraphrasing still reduces surface-form fingerprinting (single-paraphraser d≈2.19 → multi-model d≈1.5–1.8; cross-paraphraser std ≈0.06).
- Comprehensive steering tests show **scenario- and sign-dependent** behaviour:
  - Full vector: wallet compliance rises at α≈+4 (Δ≈+0.31, p<0.01); tea flat; Myanmar misinformation decreases (Δ≈−0.15).
  - Clipped vector: wallet/tea compliance rises mainly for negative α (Δ≈+0.42, p≈1e-4–7e-2); Myanmar misinformation increases at α≈+4 (Δ≈+0.24, p≈0.011).
- Effects require large |α| and exhibit high variance, so the vectors remain better suited for diagnostics than reliable control.

Legacy single-paraphraser analyses are summarized in `docs/legacy_findings.md` for historical context.
