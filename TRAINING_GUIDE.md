# Finnish GEC Training Guide

## Quick Start

### 1. Create Train/Test Data

```bash
# Use the complete pipeline (recommended)
make revita-pipeline

# Or manually
python scripts/revita_pipeline_split_and_augment.py \
    --strategy random \
    --max-error-rate 0.2 \
    --test-ratio 0.2
```

This will create:
- `data/revita/splits/train_raw.jsonl` - Raw training examples
- `data/revita/splits/test_raw.jsonl` - Raw test examples
- `data/revita/train_augmented_*.jsonl` - Augmented training data
- `data/revita/test_augmented_*.jsonl` - Augmented test data

### 2. Train Models

**Option A: ByT5-small** (Recommended for Finnish)
```bash
python scripts/train.py configs/byt5_small.yaml
```

**Option B: mT5-base** (Good baseline)
```bash
python scripts/train.py configs/mt5_base.yaml
```

**Option C: mBART-large** (Larger model, needs more GPU)
```bash
python scripts/train.py configs/mbart_large.yaml
```

### 3. Compare Results

```bash
python scripts/compare_experiments.py
```

## Model Comparison

| Model | Size | Best For | GPU Memory | Training Time |
|-------|------|----------|------------|---------------|
| **ByT5-small** | 300M | Finnish morphology | ~8GB | Medium |
| **mT5-base** | 580M | General multilingual | ~12GB | Medium |
| **mBART-large** | 610M | Larger capacity | ~16GB | Slow |

## Configuration Structure

Each config file (`configs/*.yaml`) contains:

```yaml
model:
  type: byt5                          # Model type
  pretrained_model: google/byt5-small # HuggingFace model
  task_prefix: "grammar: "            # T5 needs prefix, BART doesn't
  max_source_length: 256              # Input length
  max_target_length: 256              # Output length

data:
  train_path: data/revita/train.jsonl
  val_path: data/revita/val.jsonl
  use_weights: true                   # Use training_weight for balance

training:
  output_dir: experiments/byt5-small
  num_epochs: 5
  batch_size: 4
  learning_rate: 5e-5
  # ... more training params
```

## Key Features

### 1. Weighted Training

All configs use `use_weights: true` to apply the `training_weight` field from augmented data. This ensures:
- ✅ Balanced learning across all raw examples
- ✅ Examples with many augmentations don't dominate
- ✅ Equal contribution from simple and complex errors

### 2. Model-Specific Settings

**ByT5:**
- Uses byte-level encoding → handles Finnish morphology well
- Needs longer sequences (256 vs 128)
- Smaller batch size due to memory

**mT5:**
- Token-level encoding
- Standard sequence length (128)
- Good general-purpose baseline

**mBART:**
- No task prefix needed
- Language-specific tokens (fi_FI)
- Larger model, fewer epochs

### 3. Efficient Training

All configs use:
- **Gradient accumulation**: Effective batch size of 16
- **FP16**: Mixed precision for faster training
- **Checkpointing**: Save best model automatically
- **Early stopping**: Load best model at end

## Experiment Workflow

```bash
# 1. Clean raw data (if not already done)
python scripts/revita_clean_raw_samples.py

# 2. Split and augment (complete pipeline)
make revita-pipeline

# 3. Train multiple models in parallel
python scripts/train.py configs/byt5_small.yaml &
python scripts/train.py configs/mt5_base.yaml &

# Wait for completion...

# 4. Compare results
python scripts/compare_experiments.py
```

## Output Structure

```
experiments/
├── byt5-small/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   ├── final/                    # Best model
│   ├── trainer_state.json        # Training metrics
│   └── training_args.bin
├── mt5-base/
│   └── ...
├── mbart-large/
│   └── ...
└── comparison_results.csv        # Comparison table
```

## Customization

### Adjust Hyperparameters

Edit the YAML config:
```yaml
training:
  num_epochs: 10              # More epochs
  batch_size: 16              # Larger batch (if GPU allows)
  learning_rate: 1e-4         # Different learning rate
  warmup_steps: 1000          # More warmup
```

### Use WandB Tracking

Enable in config:
```yaml
training:
  use_wandb: true
  report_to: wandb
```

Then run:
```bash
wandb login
python scripts/train.py configs/byt5_small.yaml
```

## Troubleshooting

**Out of memory:**
- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `max_source_length`

**Training too slow:**
- Use `fp16: true`
- Increase `dataloader_num_workers`
- Use smaller model (ByT5-small vs mT5-base)

**Poor results:**
- Try more epochs
- Adjust learning rate
- Check if `use_weights: true` is enabled
- Verify train/val split is balanced

## Next Steps

After training:
1. Evaluate on test set
2. Try inference on real Finnish text
3. Fine-tune on domain-specific data
4. Experiment with different augmentation strategies
