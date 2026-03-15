# Train/Test Split Strategy for GEC Data

## Problem

Splitting data AFTER augmentation causes **data leakage**:

1. One raw example generates multiple augmented samples (different error combinations)
2. Random splitting puts augmented versions of the SAME raw text in both train and test
3. Model sees the same underlying content during training and evaluation

## Solution: Split BEFORE Augmentation

### Quick Start: Use the Pipeline

The **easiest way** to get started is to use the automated pipeline:

```bash
# Complete pipeline: split → augment train → augment test → generate EDAs
make revita-pipeline

# Or directly:
python scripts/revita_pipeline_split_and_augment.py
```

This single command:
1. ✅ Splits raw data into train/test (stratified by error count)
2. ✅ Augments training data
3. ✅ Augments test data (with different seed)
4. ✅ Generates EDA reports for both sets

### Manual Workflow (Step-by-Step)

If you prefer manual control:

```bash
# Step 1: Split raw data into train/test (BEFORE augmentation)
python scripts/revita_split_clean_raw_data.py \
  --input data/revita/exercise_errors_Finnish_cleaned.jsonl \
  --output-dir data/revita/splits \
  --test-ratio 0.2 \
  --stratify error_count \
  --seed 42

# Or use Makefile:
make revita-split

# This creates:
# - data/revita/splits/train_raw.jsonl (80% = ~2,959 examples)
# - data/revita/splits/test_raw.jsonl (20% = ~740 examples)

# Step 2: Augment training data
python scripts/revita_augment_raw_data.py \
  --input data/revita/splits/train_raw.jsonl \
  --output data/revita/train_augmented.jsonl \
  --strategy random \
  --max-error-rate 0.2 \
  --seed 42

# Or: make revita-augment-train

# Step 3: Augment test data (using SAME augmentation parameters)
python scripts/revita_augment_raw_data.py \
  --input data/revita/splits/test_raw.jsonl \
  --output data/revita/test_augmented.jsonl \
  --strategy random \
  --max-error-rate 0.2 \
  --seed 43  # Different seed to avoid identical augmentations

# Or: make revita-augment-test
```

### Why This Works

✅ **No data leakage**: Raw examples in train never appear in test
✅ **Consistent augmentation**: Same strategy applied to both sets
✅ **Representative test set**: Stratification ensures similar distributions

## Pipeline Options

### Available Make Targets

```bash
# Complete pipelines
make revita-pipeline              # Default: random sampling, stratified split
make revita-pipeline-exhaustive   # Exhaustive augmentation with limit
make revita-pipeline-test         # Quick test with limited samples

# Individual steps
make revita-split                 # Split raw data only
make revita-augment-train         # Augment training data only
make revita-augment-test          # Augment test data only

# Analysis
make revita-eda-raw               # EDA for raw cleaned data
make revita-eda-augmented         # EDA for augmented data
```

### Pipeline Parameters

```bash
python scripts/revita_pipeline_split_and_augment.py \
  --test-ratio 0.2 \              # Test set size (default: 20%)
  --stratify error_count \        # Stratification strategy
  --strategy random \             # Augmentation: random or exhaustive
  --max-error-rate 0.2 \         # Max error density (default: 20%)
  --max-augmentation 500 \       # Limit samples per raw example (optional)
  --seed 42 \                    # Random seed for reproducibility
  --skip-eda                     # Skip EDA report generation (optional)
```

## Stratification Strategies

### 1. Simple Random Split (`--stratify none`)
**When to use**: Default choice, simplest approach

```bash
python scripts/revita_split_clean_raw_data.py --stratify none --test-ratio 0.2
```

**Pros**: Simple, unbiased
**Cons**: May have slight imbalance in error counts/lengths

### 2. Stratified by Error Count (`--stratify error_count`) ⭐ **RECOMMENDED**
**When to use**: Ensure balanced error complexity in train/test

```bash
python scripts/revita_split_clean_raw_data.py --stratify error_count --test-ratio 0.2
```

**Pros**: Both sets have similar error count distributions
**Cons**: May not balance other factors

**EDA Insight**: Your data has:
- 40.5% with 1 error
- 21.9% with 2 errors
- 13.2% with 3 errors

Stratification ensures these proportions are preserved in both train/test.

### 3. Stratified by Snippet Length (`--stratify snippet_length`)
**When to use**: Ensure balanced text lengths

```bash
python scripts/revita_split_clean_raw_data.py --stratify snippet_length --test-ratio 0.2
```

**Pros**: Both sets have similar snippet length distributions
**Cons**: May not balance error complexity

**EDA Insight**: Your data has:
- Mean length: 20.4 words
- Median: 19 words
- Range: 10-99 words

### 4. Stratified by Error Density (`--stratify error_density`)
**When to use**: Ensure balanced error density (errors per word)

```bash
python scripts/revita_split_clean_raw_data.py --stratify error_density --test-ratio 0.2
```

**Pros**: Both sets have similar error rates
**Cons**: More complex stratification

**EDA Insight**: Average error density is 14.2%

## Recommendation

### For Your Dataset

Based on the EDA report, I recommend:

```bash
# Use the pipeline (easiest)
make revita-pipeline

# Or manually with stratified split by error count
python scripts/revita_split_clean_raw_data.py \
  --stratify error_count \
  --test-ratio 0.2 \
  --seed 42
```

**Why error_count stratification?**

1. ✅ Your data has clear error count distribution (40% have 1 error, 22% have 2, etc.)
2. ✅ Ensures test set has representative samples across all error complexities
3. ✅ Augmentation already handles varying error densities via `max_error_rate`
4. ✅ Snippet lengths are relatively uniform (mean ~20 words), so no need to stratify by length

**Simple random split is also fine** if you prefer simplicity - with 3,699 examples, random sampling should give decent balance.

## Balancing Test Data

### Do we need explicit balancing?

**Short answer: NO, stratification is sufficient**

**Why:**

1. **Stratified split already balances** the key factor (error count)
2. **Augmentation creates diversity**: Each raw example generates multiple samples with different error counts (1 to max_error_rate * snippet_length)
3. **Test set size (20% = ~740 examples)** is large enough to be representative

### If you still want additional balancing

You could balance test data by:
- Equal number of samples per error count
- Equal number of samples per snippet length bucket
- Equal representation of error types (spelling, case, compound, etc.)

However, this is **NOT RECOMMENDED** because:
- Natural distribution is more realistic for evaluation
- Over-balancing can create artificial test sets that don't reflect real data
- Stratification already ensures adequate representation

## Validation Set

If you want a separate validation set for hyperparameter tuning:

```bash
# Split raw data into train/val/test (60/20/20)
# Step 1: Split off test set (20%)
python scripts/revita_split_clean_raw_data.py \
  --input data/revita/exercise_errors_Finnish_cleaned.jsonl \
  --output-dir data/revita/splits \
  --test-ratio 0.2 \
  --stratify error_count \
  --seed 42

# Step 2: Split train into train/val (75/25 of remaining 80% = 60/20 overall)
python scripts/revita_split_clean_raw_data.py \
  --input data/revita/splits/train_raw.jsonl \
  --output-dir data/revita/splits \
  --test-ratio 0.25 \
  --stratify error_count \
  --seed 43

# Rename outputs
mv data/revita/splits/train_raw.jsonl data/revita/splits/train_raw_final.jsonl
mv data/revita/splits/test_raw.jsonl data/revita/splits/val_raw.jsonl
```

This gives you:
- Train: 60% (~2,219 raw examples)
- Validation: 20% (~740 raw examples)
- Test: 20% (~740 raw examples)

## Summary

### Key Principles

1. ✅ **Split BEFORE augmentation** to prevent data leakage
2. ✅ **Use stratification** to ensure balanced distributions
3. ✅ **Apply same augmentation** to train/test (but different seeds)
4. ✅ **20% test ratio** is standard and sufficient
5. ✅ **Natural distribution** in test set is better than over-balancing

### Recommended Commands

```bash
# Easiest: Use the complete pipeline
make revita-pipeline

# Or step-by-step with Make targets
make revita-split
make revita-augment-train
make revita-augment-test

# Or manual with Python scripts
python scripts/revita_pipeline_split_and_augment.py
```

This ensures:
- No raw examples overlap between train/test
- Similar error complexity in both sets
- Representative test set for evaluation

## Script Reference

All scripts follow consistent naming: `revita_<action>_<target>_data.py`

- [`revita_split_clean_raw_data.py`](../scripts/revita_split_clean_raw_data.py) - Split raw data
- [`revita_augment_raw_data.py`](../scripts/revita_augment_raw_data.py) - Augment raw data
- [`revita_pipeline_split_and_augment.py`](../scripts/revita_pipeline_split_and_augment.py) - Complete pipeline
- [`revita_eda_cleaned_raw_data.py`](../scripts/revita_eda_cleaned_raw_data.py) - EDA for raw data
- [`revita_eda_augmented_data.py`](../scripts/revita_eda_augmented_data.py) - EDA for augmented data
