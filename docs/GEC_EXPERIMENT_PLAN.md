# GEC Model Comparison Experiment Plan

## Overview

Comprehensive plan for training and comparing multiple seq2seq models on Finnish GEC using Revita data.

---

## 1. Evaluation Metrics

### Primary Metrics for GEC

#### **F0.5 Score** (Precision-Weighted) ⭐ **PRIMARY METRIC**

- **Why F0.5?** In GEC, precision is more important than recall (better to miss corrections than introduce errors)
- **Implementation Options:**
  1. **ERRANT** (Error Annotation Toolkit) - Recommended for GEC
     - Edit-based evaluation
     - Provides error type classification
     - Standard in GEC research
  2. **M2 Scorer** (MaxMatch) - Alternative
     - Phrase-level matching
     - Used in CoNLL shared tasks

**F0.5 Formula:**
```
F0.5 = (1 + 0.5²) × (Precision × Recall) / (0.5² × Precision + Recall)
     = 1.25 × (P × R) / (0.25P + R)
```

#### **GLEU** (Generalized Language Evaluation Understanding)

- Variant of BLEU specifically designed for GEC
- Sentence-level metric (vs. corpus-level BLEU)
- Considers both correctness and fluency

#### **Exact Match Accuracy**

- Simple baseline: % of perfectly corrected sentences
- Easy to understand but harsh metric

#### **Edit Distance Metrics**

- **Levenshtein Distance**: Character-level edits
- **Word Error Rate (WER)**: Word-level edits
- Useful for understanding model behavior

### Secondary Metrics

- **BLEU**: Reference-based similarity (less suitable for GEC but widely known)
- **Perplexity**: Model confidence
- **Inference Time**: Practical deployment consideration

### Stratified Evaluation

Break down metrics by:
1. **Error Count**: 1 error, 2 errors, 3+ errors
2. **Error Type** (if annotated): Spelling, grammar, morphology, etc.
3. **Sentence Length**: Short (<15 words), medium (15-30), long (>30)
4. **Error Density**: Low (<10%), medium (10-20%), high (>20%)

---

## 2. Models to Compare

### Seq2Seq Models (Your Current Focus)

| Model | Size | Strengths | Considerations |
|-------|------|-----------|----------------|
| **ByT5-small** | 300M | ⭐ **Best for Finnish morphology** (byte-level) | Higher memory, longer sequences |
| **mT5-base** | 580M | Good multilingual baseline | Standard choice |
| **mT5-small** | 300M | Faster, less memory | Lower capacity |
| **mBART-large-50** | 610M | Strong multilingual model | Needs more resources |
| **BART-base** | 140M | Faster baseline (English-focused) | May underperform on Finnish |

### Recommended Priority

1. **ByT5-small** ← Start here (best for Finnish)
2. **mT5-base** ← Strong baseline
3. **mBART-large-50** ← If you have GPU memory
4. **mT5-small** ← For comparison

### Other Approaches (Future Work)

- **GECToR** (Token classification): Edit-based, faster inference
- **LLM with LoRA**: GPT-style models
- **Copy Mechanism**: Explicit copy attention
- **Multitask GED+GEC**: Joint detection and correction

---

## 3. Experiment Design

### Training Configuration

All models use:
- ✅ **Weighted training** (from `training_weight` field)
- ✅ **Same data splits** (train/test from pipeline)
- ✅ **Consistent hyperparameters** (where possible)
- ✅ **Early stopping** (on validation loss)
- ✅ **Best model selection**

### Data Setup

```bash
# Use the augmented data from pipeline
Train: data/revita/splits/train_augmented_random_greedy_errdensity20_clean_seed42.jsonl
Test:  data/revita/splits/test_augmented_random_greedy_errdensity20_clean_seed42.jsonl

# Split train into train/val (90/10)
# OR use test set for both validation and final eval (simpler)
```

### Hyperparameter Settings

**Consistent Across Models:**
- Effective batch size: 16 (via gradient accumulation)
- Learning rate: 5e-5 (3e-5 for larger models)
- Warmup steps: 500
- Max epochs: 5 (3 for larger models)
- FP16: Enabled (for speed)

**Model-Specific:**
- Batch size: Adjusted for memory (4-8)
- Sequence length: Adjusted for model (128-256)

---

## 4. Implementation Steps

### Step 1: Implement Evaluation Metrics ⚠️ **TODO**

The current `src/evaluate.py` has TODOs for metrics. You need to implement:

```python
# Option A: Use ERRANT (Recommended)
pip install errant

# Option B: Use M2 Scorer
git clone https://github.com/nusnlp/m2scorer

# Option C: Implement simple GLEU yourself
# (Formula available in GEC papers)
```

**Priority:** Implement at least one of:
1. **ERRANT-based F0.5** (best, most standard)
2. **Simple Exact Match** (baseline, easy)
3. **GLEU** (good middle ground)

### Step 2: Update Config Files

Your configs reference old paths. Update them:

```yaml
data:
  train_path: data/revita/splits/train_augmented_random_greedy_errdensity20_clean_seed42.jsonl
  val_path: data/revita/splits/test_augmented_random_greedy_errdensity20_clean_seed42.jsonl  # Use test as val for now
  test_path: data/revita/splits/test_augmented_random_greedy_errdensity20_clean_seed42.jsonl
```

Or create a train/val split:
```bash
# Split training data 90/10
head -n 62333 data/revita/splits/train_augmented*.jsonl > data/revita/splits/train.jsonl
tail -n 6926 data/revita/splits/train_augmented*.jsonl > data/revita/splits/val.jsonl
```

### Step 3: Train Models in Parallel

```bash
# Train all models concurrently (if you have multiple GPUs)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py configs/byt5_small.yaml &
CUDA_VISIBLE_DEVICES=1 python scripts/train.py configs/mt5_base.yaml &
CUDA_VISIBLE_DEVICES=2 python scripts/train.py configs/mbart_large.yaml &

# Or sequentially
make train-byt5
make train-mt5
make train-mbart
```

### Step 4: Generate Predictions

```bash
# For each model, generate predictions on test set
python src/generate_predictions.py \
  --model experiments/byt5-small/best \
  --test data/revita/splits/test_augmented_*.jsonl \
  --output predictions/byt5-small.jsonl
```

### Step 5: Evaluate All Models

```bash
# Evaluate each model
python src/evaluate.py \
  --predictions predictions/byt5-small.jsonl \
  --test data/revita/splits/test_augmented_*.jsonl \
  --output results/byt5-small.json

# Repeat for all models...
```

### Step 6: Compare Results

```bash
# Aggregate comparison
python scripts/compare_experiments.py \
  --results results/*.json \
  --output comparison.csv
```

---

## 5. Recommended Workflow

### Quick Start (Recommended)

```bash
# 1. Implement basic evaluation metric first
#    (Start with Exact Match or simple GLEU)

# 2. Create train/val split
python -c "
import json
from pathlib import Path

train_file = Path('data/revita/splits/train_augmented_random_greedy_errdensity20_clean_seed42.jsonl')
samples = []
with open(train_file) as f:
    samples = [line for line in f]

split = int(len(samples) * 0.9)
with open('data/revita/splits/train.jsonl', 'w') as f:
    f.writelines(samples[:split])
with open('data/revita/splits/val.jsonl', 'w') as f:
    f.writelines(samples[split:])
"

# 3. Update all config files to use new paths
sed -i '' 's|data/revita/train.jsonl|data/revita/splits/train.jsonl|g' configs/*.yaml
sed -i '' 's|data/revita/val.jsonl|data/revita/splits/val.jsonl|g' configs/*.yaml

# 4. Train ByT5 first (best for Finnish)
python scripts/train.py configs/byt5_small.yaml

# 5. Evaluate on test set
python src/evaluate.py \
  --config configs/byt5_small.yaml \
  --split test

# 6. If ByT5 works well, train others
python scripts/train.py configs/mt5_base.yaml
python scripts/train.py configs/mbart_large.yaml

# 7. Compare all models
python scripts/compare_experiments.py
```

### Full Experiment (Comprehensive)

1. **Week 1: Setup & Baseline**
   - Implement evaluation metrics (F0.5 + GLEU)
   - Train ByT5-small (best for Finnish)
   - Establish baseline performance

2. **Week 2: Model Comparison**
   - Train mT5-base, mT5-small
   - Train mBART-large (if resources permit)
   - Compare all seq2seq models

3. **Week 3: Analysis & Optimization**
   - Error analysis by type/length/density
   - Hyperparameter tuning for best model
   - Ablation studies (with/without weights, different error rates)

4. **Week 4: Final Evaluation**
   - Ensemble methods (if beneficial)
   - Human evaluation on sample
   - Final report and model selection

---

## 6. Expected Results

### Baseline Expectations

Based on GEC research, typical performance on learner data:

| Metric | Conservative | Optimistic |
|--------|-------------|-----------|
| **F0.5** | 30-40% | 50-60% |
| **GLEU** | 60-70 | 75-85 |
| **Exact Match** | 20-30% | 40-50% |

### Model Ranking (Expected)

1. **ByT5-small** ← Likely best for Finnish
2. **mT5-base** ← Strong second
3. **mBART-large** ← Similar to mT5
4. **mT5-small** ← Faster but lower quality

**Why ByT5 should win:** Finnish is morphologically rich, byte-level encoding handles inflections better than subword tokenization.

---

## 7. Critical TODOs

### Must-Do Before Training

- [ ] **Implement F0.5 metric** (use ERRANT or M2 scorer)
- [ ] **Implement GLEU metric**
- [ ] **Create train/val split** from training data
- [ ] **Update all config files** with correct data paths
- [ ] **Write prediction generation script**

### Nice-to-Have

- [ ] Implement per-error-type evaluation
- [ ] Add confidence calibration metrics
- [ ] Implement ensemble methods
- [ ] Add human evaluation framework

---

## 8. Metrics Implementation Guide

### Option 1: Use ERRANT (Recommended)

```python
# Install ERRANT
pip install errant

# Example usage
import errant

annotator = errant.load('en')  # Or 'fi' if available

# Annotate edits
orig = annotator.parse('This are wrong')
cor = annotator.parse('This is correct')
edits = annotator.annotate(orig, cor)

# Compute F0.5
from errant.commands.compare_m2 import evaluate_edits
precision, recall, f05 = evaluate_edits(gold_edits, pred_edits, beta=0.5)
```

### Option 2: Simple GLEU

```python
def gleu_score(prediction, reference, source):
    """Simple GLEU implementation."""
    # 1. Get n-gram matches between pred and ref
    # 2. Get n-gram matches between pred and src
    # 3. Penalize copying from source
    # 4. Reward matching reference

    # (Implementation details in GEC papers)
    pass
```

### Option 3: Exact Match (Simplest Start)

```python
def exact_match_accuracy(predictions, references):
    """Simple exact match accuracy."""
    matches = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    return matches / len(predictions)
```

---

## 9. Summary

### The Plan

1. ✅ **Data is ready** (train/test splits from pipeline)
2. ⚠️ **Implement metrics** (F0.5, GLEU, Exact Match)
3. 🔧 **Update configs** (fix data paths)
4. 🚀 **Train models** (ByT5 → mT5 → mBART)
5. 📊 **Evaluate & compare** (stratified analysis)
6. 🎯 **Select best model** for deployment

### Quick Start Command

```bash
# After implementing metrics and updating configs:
make train-byt5     # Start with best-for-Finnish model
make eval           # Evaluate on test set
```

### Next Immediate Steps

1. **Implement evaluation metrics** (start with Exact Match, then add F0.5)
2. **Create train/val split** from training data
3. **Update config files** with correct paths
4. **Train ByT5-small** as baseline
5. **Iterate from there**

---

**Ready to start?** The data is prepared, the framework is in place, you just need to:
1. Implement the evaluation metrics
2. Update the configs
3. Train & compare models!
