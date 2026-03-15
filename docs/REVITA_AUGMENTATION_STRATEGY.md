# Revita Data Augmentation Strategy

## Overview

This document describes our strategy for augmenting Revita learner error data into training samples for Finnish GEC models, addressing key challenges around balanced representation and avoiding bias.

## Problem Statement

### The Raw Data Structure

Revita data format:
```json
{
  "snippet": ["Suomi", " ", "on", " ", "kaunis", " ", "maa", "."],
  "errors": [
    {
      "wid": [4],
      "word": "kaunis",
      "instances": ["kauniis", "kaunista", "kaunii"]
    }
  ]
}
```

Each raw example contains:
- **snippet**: Correct text (tokenized)
- **errors**: Error targets with multiple incorrect instances

### The Augmentation Challenge

From each raw example, we generate multiple training samples by:
1. Selecting N errors (from 1 to max)
2. Picking instances for each error
3. Applying errors to create (corrupted, correct) pairs

**Key Issue**: Examples with different error counts generate vastly different numbers of samples:

```
Raw Example A (3 total errors):
  → Generates ~27 samples (exhaustive)
  → Correct text appears 27 times in training data

Raw Example B (10 total errors):
  → Generates 500 samples (hits limit)
  → Correct text appears 500 times in training data

Problem: Example B dominates training 18× more than Example A!
```

## Our Solution: Frequency-Weighted Training

### Core Principle

**During augmentation**: Generate samples freely (exhaustive or random)
**During training**: Weight each sample inversely to its correct text frequency

```python
training_weight = 1.0 / frequency_of_correct_text
```

This ensures **every unique raw example contributes equally** to model training.

### Mathematical Balance

For a correct text appearing `n` times in augmented data:
- Each occurrence gets weight `w = 1/n`
- Total contribution: `n × (1/n) = 1.0`

Result: All 3,699 raw examples have **equal influence** on the model!

## Implementation

### Step 1: Augmentation with Metadata

During `revita_augment_raw_data.py`, we add frequency tracking:

```json
{
  "corrupted": "Suomi on kauniis maa.",
  "correct": "Suomi on kaunis maa.",
  "num_errors": 1,
  "snippet_length": 18,
  "error_rate": 0.056,
  "edits": [...],
  "training_weight": 0.037,           // 1.0 / 27
  "correct_frequency": 27             // This correct text appears 27 times
}
```

### Step 2: Weighted Loss in Training

Modify your training loop to use `training_weight`:

```python
# Load batch
batch = [json.loads(line) for line in batch_lines]

corrupted = [s['corrupted'] for s in batch]
correct = [s['correct'] for s in batch]
weights = torch.tensor([s['training_weight'] for s in batch])

# Forward pass
loss = model(corrupted, correct)

# Apply weights to balance contributions
weighted_loss = (loss * weights).mean()

# Backprop
weighted_loss.backward()
```

## Benefits

### 1. Balanced Representation
✅ Every raw example contributes equally
✅ No overfitting to high-error examples
✅ Simple examples get proper attention

### 2. Maximum Data Utilization
✅ Keep all augmented samples (no filtering)
✅ Maintain exhaustive single-error coverage
✅ Preserve error combination diversity

### 3. Transparency
✅ Frequency metadata shows imbalance
✅ Easy to analyze and debug
✅ Can monitor weight distribution

## Augmentation Strategy Details

### Exhaustive Sampling

Generate all combinations of errors and instances:

```python
Strategy: exhaustive
Max per raw example: 500

Example with 3 errors:
  1 error:  C(3,1) × 2 = 6 samples
  2 errors: C(3,2) × 4 = 12 samples
  3 errors: C(3,3) × 8 = 8 samples
  Total: 26 samples ✅ All generated

Example with 20 errors:
  1 error:  C(20,1) × 2 = 40 samples
  2 errors: C(20,2) × 4 = 760 samples
  Stops at 500 samples ⚠️ Partial coverage
```

### Coverage Guarantees

With exhaustive strategy (limit=500):
- ✅ **100% coverage** of all single-error cases
- ✅ **Complete coverage** for examples with ≤5 errors
- ✅ **Systematic coverage** for examples with 6-10 errors
- ⚠️ **Partial coverage** for examples with 10+ errors

### Error Density Control

```python
min_error_rate = 0.05  # Min 5% of snippet
max_error_rate = 0.30  # Max 30% of snippet

For 18-word snippet:
  min_errors = 1
  max_errors = 5
```

## Statistics & Monitoring

### Augmentation Output

During augmentation, you'll see:

```
⚖️  Calculating training weights...
   Unique correct texts: 3,699
   Frequency range: Min=12, Max=500, Avg=135.5
   Most repeated text: 500× (weight=0.002)
   Least repeated text: 12× (weight=0.083)
```

### Interpretation

- **Unique correct texts = 3,699**: Matches raw example count ✅
- **Frequency range**: Shows augmentation imbalance
  - Max 500 → Hit the per-example limit
  - Min 12 → Small examples with few errors
- **Average ~135**: Typical expansion factor per raw example

### Weight Distribution Analysis

After augmentation, analyze the distribution:

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load and analyze
frequencies = []
weights = []

with open('train.jsonl') as f:
    for line in f:
        s = json.loads(line)
        frequencies.append(s['correct_frequency'])
        weights.append(s['training_weight'])

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(frequencies, bins=50)
ax1.set_xlabel('Correct Text Frequency')
ax1.set_ylabel('Count')
ax1.set_title('How many times each correct text appears')

ax2.hist(weights, bins=50)
ax2.set_xlabel('Training Weight')
ax2.set_ylabel('Count')
ax2.set_title('Distribution of training weights')

plt.savefig('augmentation_balance_analysis.png')
```

## Usage

### Generate Augmented Data

```bash
# Recommended: Use the complete pipeline
make revita-pipeline

# Or for exhaustive strategy with limit:
make revita-pipeline-exhaustive

# Or manually:
python scripts/revita_pipeline_split_and_augment.py \
  --strategy exhaustive \
  --max-augmentation 500
```

**Note**: The pipeline automatically handles splitting raw data before augmentation, preventing data leakage between train and test sets.

### Train with Weights

Update your training code to use `training_weight` field from JSONL:

```python
# In your training loop
for batch in dataloader:
    # Extract weights
    weights = batch['training_weight']

    # Compute loss
    loss = criterion(outputs, targets)

    # Apply weights
    weighted_loss = (loss * weights).mean()

    # Optimize
    weighted_loss.backward()
    optimizer.step()
```

## Validation

### Sanity Checks

1. **Total weight per unique correct text = 1.0**:
   ```python
   from collections import defaultdict

   snippet_total_weight = defaultdict(float)
   for sample in samples:
       snippet_total_weight[sample['correct']] += sample['training_weight']

   # All should be ~1.0
   for correct_text, total_weight in snippet_total_weight.items():
       assert 0.99 < total_weight < 1.01
   ```

2. **Weight × Frequency = 1.0**:
   ```python
   for sample in samples:
       computed = sample['training_weight'] * sample['correct_frequency']
       assert abs(computed - 1.0) < 0.001
   ```

## Expected Training Behavior

### With Balanced Weights ✅
- Smooth loss curves (all examples contribute equally)
- Better performance on rare error patterns
- More robust generalization
- May converge slightly slower initially

### Without Weights ❌
- Loss dominated by high-frequency examples
- Overfitting to complex multi-error patterns
- Poor performance on simple errors
- Faster initial convergence but worse final performance

## Alternative Strategies Considered

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Cap samples per example** | Simple, predictable size | Wastes training data | ❌ Rejected |
| **Stratified by error count** | Balanced groups | Arbitrary grouping, complex | ❌ Rejected |
| **Frequency weighting** | No data loss, perfect balance | Needs training code change | ✅ **Chosen** |
| **Filter duplicates** | Removes repetition | Loses error diversity | ❌ Rejected |

## Future Enhancements

Potential improvements to explore:

1. **Error-type weighting**: Further balance by error category (spelling vs case vs compounds)
2. **Dynamic weighting**: Adjust during training based on model performance per error type
3. **Curriculum learning**: Start uniform, gradually increase weighting
4. **Length normalization**: Adjust for snippet length differences

## References

- Inverse frequency weighting similar to TF-IDF in information retrieval
- Class balancing in imbalanced classification
- Importance sampling in RL and active learning

---

**Created**: 2026-03-14
**Version**: 1.0
**Status**: Production Ready
