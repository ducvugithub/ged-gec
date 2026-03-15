# Finnish GEC — Grammatical Error Correction

Research project for Finnish grammatical error correction using multiple approaches: seq2seq models (T5, mBART, BART), token classification (GECToR-style), copy mechanisms, LLM prompting, and multitask GED+GEC.

## Overview

This project trains and evaluates GEC models for Finnish using:
- **Synthetic data**: UD Finnish treebanks with linguistically-aware corruption
- **Real learner data**: Revita platform data (reserved for evaluation)

### Training Strategy

1. **Full train on synthetic data** — primary training stage on synthetically corrupted Finnish sentences
2. **Optional fine-tune on Revita** — only if data volume permits after reserving evaluation set
3. **Evaluate on Revita held-out set** — stratified by error type frequency

## Approaches

1. **Seq2Seq** — T5, mT5, mBART, BART
2. **Token Classification** — GECToR-style with XLM-RoBERTa
3. **Copy Mechanism** — mT5/mBART with explicit copy attention
4. **LLM Fine-tuning** — LoRA/QLoRA on Llama/Mistral
5. **Multitask GED+GEC** — Joint error detection and correction

## Quick Start

### Setup
```bash
pip install -r requirements.txt
```

### 1. Generate Synthetic Data
```bash
# Confusion sets (primary strategy)
python -m src.synthetic_generation.confusion_sets --input data/ud_treebank.txt --output data/synthetic/confusion_sets/

# LLM-generated errors (high quality)
python -m src.synthetic_generation.llm_errors --input data/ud_treebank.txt --output data/synthetic/llm_generated/
```

### 2. Error Analysis (run first!)
```bash
python -m src.error_analysis --data data/synthetic/confusion_sets/train.jsonl
```

### 3. Train a Model
```bash
# mT5 seq2seq
python -m src.train --config configs/seq2seq_mt5.yaml

# GECToR token classification
python -m src.train --config configs/gector_xlmr.yaml

# Multitask GED+GEC
python -m src.train --config configs/multitask.yaml
```

### 4. Evaluate
```bash
python -m src.evaluate --config configs/seq2seq_mt5.yaml --split revita_test
```

## Data Structure

```
data/
├── synthetic/          # Generated (corrupted, corrected) pairs
│   ├── confusion_sets/
│   ├── random_ops/
│   ├── back_translated/
│   └── llm_generated/
└── revita/             # Real learner data (reserved for evaluation)
```

## Models

All model implementations are in `src/models/`:
- `seq2seq/` — T5, mT5, mBART, BART trainers
- `gector/` — Token classification with edit operations
- `copy/` — Copy mechanism augmented encoders
- `llm/` — LoRA fine-tuning and prompting
- `multitask/` — Joint GED+GEC architecture

## Evaluation Metrics

- **F0.5** — primary metric (precision-weighted)
- **GLEU** — generalized language evaluation
- **Per-error-type stratification** — morphological, syntactic, lexical, agreement, punctuation

## Citation

If you use this code, please cite the original GECToR paper and relevant model papers (mT5, mBART, etc.).

## License

None
