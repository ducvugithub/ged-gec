#!/bin/bash
# Generate full augmented dataset with exhaustive strategy

echo "🚀 Starting full dataset augmentation..."
echo "   Strategy: Exhaustive"
echo "   Limit: 500 samples per example"
echo "   Expected output: ~1.85M samples"
echo ""

python scripts/augment_revita_data.py \
  --input data/revita/exercise_errors_Finnish.jsonl \
  --output data/revita/augmented_Finnish_exhaustive.jsonl \
  --strategy exhaustive \
  --max-augmentation-per-raw-example 500 \
  --create-split \
  --val-ratio 0.1

echo ""
echo "✅ Done! Check data/revita/ for train.jsonl and val.jsonl"
