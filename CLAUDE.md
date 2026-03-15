# Finnish GEC — Experiment Plan

## Overview

### Training Strategy

Given the limited size of real learner data (Revita), the pipeline is:

1. **Full train on synthetic data** — each model trained from scratch (or from pretrained LM weights) on synthetically corrupted Finnish sentences. This is the primary training stage.
2. **Optional fine-tune on Revita** — only if Revita volume is sufficient after reserving a held-out evaluation set. If data is too limited, Revita is used exclusively for evaluation.
3. **Evaluate on Revita held-out set** — stratified by error type frequency.

> ⚠️ Revita data is scarce. Default assumption: reserve it entirely for evaluation unless volume analysis shows enough headroom for a fine-tuning split.

---

## Synthetic Data

### Source Text

- **Universal Dependencies Finnish treebanks** — clean, grammatically correct Finnish sentences. A UD-based morphological analyser is available and should be used to guide linguistically-aware corruption.
- Supplement with Finnish Wikipedia or news corpora for domain diversity if needed.

### Corruption Approaches

The following strategies can be used independently or combined to generate (corrupted, corrected) pairs:

| Approach | Description | Quality | Cost |
|---|---|---|---|
| **Confusion sets** | Per token, build a set of plausible confusable forms (e.g. partitive vs. accusative, past vs. present) using the UD analyser and randomly substitute | Medium-High | Low-Medium |
| **Random character/token ops** | Insertions, deletions, swaps at character or token level | Low | Very low |
| **Back-translation** | Finnish → X → Finnish via MT; surface differences become errors | Medium | Medium |
| **LLM-generated errors** | Prompt an LLM to introduce realistic L1-specific learner errors into correct sentences | High | High |

> 🔑 Confusion sets are the primary strategy — they leverage the UD analyser directly and produce linguistically grounded errors. LLM-generated errors are the quality ceiling worth exploring once the confusion set pipeline is stable. Random ops and back-translation are cheap supplements for volume and diversity.

### Recommended Pipeline

1. Use UD treebank sentences as clean source
2. Apply rule-based + confusion set corruption as primary noise
3. Optionally layer LLM-generated errors for a higher-realism synthetic split
4. Track which corruption strategy produced each example — useful for ablations

---

## Approaches

### 1. Seq2Seq — T5 / mT5

- **Models**: `google/mt5-base`, `google/mt5-large`
- **Input**: corrupted sentence → **Target**: corrected sentence
- **Trainer**: `src/models/seq2seq/trainer.py`
- **Notes**:
  - mT5 preferred over vanilla T5 for Finnish morphology coverage
  - Watch for copy bias vs. overcorrection tradeoff
  - Warm-start from mT5 pretrained weights before synthetic training

---

### 2. Seq2Seq — mBART

- **Models**: `facebook/mbart-large-cc25`, `facebook/mbart-large-50`
- **Input/Target**: sentences with `fi_FI` language tag
- **Trainer**: `src/models/seq2seq/trainer.py` (shared with T5/mT5)
- **Notes**:
  - Denoising pretraining objective aligns naturally with GEC
  - May generalize better across error types due to pretraining diversity

---

### 3. Seq2Seq — BART

- **Models**: `facebook/bart-base`, `facebook/bart-large`
- **Trainer**: `src/models/seq2seq/trainer.py` (shared)
- **Notes**:
  - English-centric — treat as ablation vs. mBART
  - Useful to isolate effect of multilingual pretraining

---

### 4. Token Classification — GECToR-style

- **Encoder**: `xlm-roberta-base`, `xlm-roberta-large`
- **Head**: per-token classification over edit operations `{KEEP, DELETE, REPLACE→X, INSERT→X}`
- **Inference**: iterative — run multiple passes until no edits predicted
- **Trainer**: `src/models/gector/trainer.py`
- **Notes**:
  - Faster inference and more interpretable than seq2seq
  - Edit label vocabulary design is critical — Finnish morphological space is large, fixed vocab may bottleneck open-vocabulary replacements
  - UD analyser can help define a principled label set

---

### 5. Encoder-Decoder with Copy Mechanism

- **Base**: mT5 or mBART augmented with explicit copy attention / pointer network
- **Trainer**: `src/models/copy/trainer.py`
- **Notes**:
  - Biases toward preserving input tokens, only generates when correction needed
  - Directly addresses overcorrection problem
  - Important for conservative correction in learner-facing feedback (Revita use case)

---

### 6. LLM Prompting / Instruction Fine-tuning

- **Models**: `meta-llama/Llama-3-8B-Instruct`, `mistralai/Mistral-7B-Instruct`
- **Modes**:
  - Few-shot prompting (no training) — use as sanity check upper bound
  - LoRA/QLoRA fine-tuning on synthetic (corrupted, corrected) pairs
- **Trainer**: `src/models/llm/trainer.py`
- **Notes**:
  - Tendency to paraphrase hurts F0.5 — include minimal-edit instruction in prompt
  - LoRA fine-tuning makes this tractable on limited GPU budget

---

### 7. Multitask GED + GEC (Joint)

- **Architecture**: shared encoder, two heads
  - **GED head**: token-level binary/multi-class error detection
  - **GEC head**: seq2seq correction
- **Trainer**: `src/models/multitask/trainer.py`
- **Notes**:
  - GED supervises encoder, GEC supervises decoder — avoids cascade errors of two-stage pipeline
  - GED output usable standalone for Revita learner feedback (flag without correcting)
  - Most complex to implement but most informative architecture

---

## Error Analysis

Run **before training any model** to understand the data distribution:

- Compute error type frequency in synthetic data and (if used for training) Revita
- Stratify test set by error type — morphological, syntactic, lexical, punctuation, agreement
- Report per-type F0.5 alongside aggregate metrics for all models
- Flag performance drop-off on low-frequency error types as generalization signal
- Use UD analyser to assist with error type annotation

---

## Data

| Split | Source | Notes |
|---|---|---|
| Synthetic train | UD Finnish treebanks + noise | Primary training data for all models |
| Optional fine-tune | Revita | Only if volume allows — check before splitting |
| Evaluation | Revita held-out | Stratified by error type, reserved by default |

---

## Repo Structure

```
finnish-gec/
├── data/
│   ├── synthetic/          # generated (corrupted, corrected) pairs
│   │   ├── confusion_sets/
│   │   ├── random_ops/
│   │   ├── back_translated/
│   │   └── llm_generated/
│   └── revita/             # real learner data — handle carefully
├── src/
│   ├── synthetic_generation/   # corruption pipeline (uses UD analyser)
│   │   ├── confusion_sets.py
│   │   ├── random_ops.py
│   │   ├── back_translation.py
│   │   └── llm_errors.py
│   ├── models/
│   │   ├── seq2seq/        # T5, mT5, mBART, BART
│   │   │   ├── model.py
│   │   │   └── trainer.py
│   │   ├── gector/         # token classification
│   │   │   ├── model.py
│   │   │   └── trainer.py
│   │   ├── copy/           # copy mechanism
│   │   │   ├── model.py
│   │   │   └── trainer.py
│   │   ├── llm/            # prompting + LoRA fine-tuning
│   │   │   ├── model.py
│   │   │   └── trainer.py
│   │   └── multitask/      # joint GED + GEC
│   │       ├── model.py
│   │       └── trainer.py
│   ├── train.py            # dispatcher — reads config, calls model-specific trainer
│   ├── evaluate.py         # shared eval: F0.5, GLEU, per-error-type stratification
│   └── error_analysis.py   # run first — distribution analysis on data
├── configs/
│   ├── seq2seq_mt5.yaml
│   ├── seq2seq_mbart.yaml
│   ├── seq2seq_bart.yaml
│   ├── gector_xlmr.yaml
│   ├── copy_mt5.yaml
│   ├── llm_lora.yaml
│   └── multitask.yaml
├── notebooks/
│   └── error_analysis.ipynb
└── README.md
```

### Train / Evaluate Flow

```
train.py --config configs/seq2seq_mt5.yaml
    └── reads model type from config
    └── calls src/models/seq2seq/trainer.py
    └── logs to wandb / output dir

evaluate.py --config configs/seq2seq_mt5.yaml --split revita_test
    └── shared across all model types
    └── outputs aggregate + per-error-type metrics
```