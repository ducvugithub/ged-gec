# Reports Directory

This directory contains generated EDA reports and analysis outputs.

## Generated Reports

- `revita_raw_cleaned_eda.md` - Comprehensive exploratory data analysis of cleaned raw Revita data
- `train_augmented_*_eda.md` - EDA for augmented training data
- `test_augmented_*_eda.md` - EDA for augmented test data

## Generating Reports

```bash
# Generate EDA for raw cleaned data
make revita-eda-raw

# Or run directly
python scripts/revita_eda_cleaned_raw_data.py

# Generate EDA for augmented data (after running pipeline)
make revita-eda-augmented

# Or run directly
python scripts/revita_eda_augmented_data.py --data-file data/revita/train_augmented.jsonl
```

## Report Contents

The EDA report includes:
1. **Basic Statistics** - dataset size, sentence lengths, field coverage
2. **Error Analysis** - error types, categories, distribution
3. **Edit Distance Analysis** - Levenshtein distances between corrupted/correct
4. **Language Patterns** - word frequencies, Finnish-specific patterns
5. **Recommendations** - actionable insights for model training