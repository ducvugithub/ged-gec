.PHONY: help install setup clean lint test

help:
	@echo "Finnish GEC — Available Commands"
	@echo ""
	@echo "  make install         Install dependencies"
	@echo "  make setup          Setup development environment"
	@echo "  make clean          Clean generated files"
	@echo "  make lint           Run linters"
	@echo "  make test           Run tests"
	@echo ""
	@echo "Data Generation:"
	@echo "  make gen-confusion  Generate confusion set corruptions"
	@echo "  make gen-random     Generate random ops corruptions"
	@echo "  make gen-bt         Generate back-translation corruptions"
	@echo "  make gen-llm        Generate LLM-based corruptions"
	@echo ""
	@echo "Data Pipeline (Revita):"
	@echo "  make revita-pipeline             Complete pipeline: split + augment train/test"
	@echo "  make revita-split                Split raw data into train/test"
	@echo "  make revita-augment-train        Augment training data only"
	@echo "  make revita-augment-test         Augment test data only"
	@echo ""
	@echo "Data Analysis (Revita):"
	@echo "  make revita-eda-raw              Generate EDA for raw cleaned data"
	@echo "  make revita-eda-augmented        Generate EDA for augmented data"
	@echo ""
	@echo "Training (Seq2Seq Models):"
	@echo "  make train-byt5     Train ByT5-small (recommended for Finnish)"
	@echo "  make train-mt5      Train mT5-base"
	@echo "  make train-mbart    Train mBART-large"
	@echo ""
	@echo "Inference (Generate Predictions):"
	@echo "  make infer-byt5     Generate predictions with ByT5-small"
	@echo "  make infer-mt5      Generate predictions with mT5-base"
	@echo "  make infer-mbart    Generate predictions with mBART-large"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval-byt5      Evaluate ByT5-small predictions"
	@echo "  make eval-mt5       Evaluate mT5-base predictions"
	@echo "  make eval-mbart     Evaluate mBART-large predictions"
	@echo "  make error-analysis Run error distribution analysis"

install:
	pip install -r requirements.txt
	pip install -e .

setup: install
	@echo "✓ Environment setup complete"
	@echo "Next steps:"
	@echo "  1. Run data pipeline: make revita-pipeline"
	@echo "  2. Check EDA reports in reports/"
	@echo "  3. Train models: make train-byt5"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf outputs/ wandb/ .pytest_cache/

lint:
	black src/ --check
	flake8 src/ --max-line-length=100
	mypy src/

test:
	pytest tests/ -v

# Data generation targets
gen-confusion:
	python -m src.synthetic_generation.confusion_sets \
		--input data/ud_treebank.txt \
		--output data/synthetic/confusion_sets/train.jsonl \
		--corruption-rate 0.15

gen-random:
	python -m src.synthetic_generation.random_ops \
		--input data/ud_treebank.txt \
		--output data/synthetic/random_ops/train.jsonl \
		--corruption-rate 0.10

gen-bt:
	python -m src.synthetic_generation.back_translation \
		--input data/ud_treebank.txt \
		--output data/synthetic/back_translated/train.jsonl \
		--pivot-lang en

gen-llm:
	python -m src.synthetic_generation.llm_errors \
		--input data/ud_treebank.txt \
		--output data/synthetic/llm_generated/train.jsonl \
		--model gpt-3.5-turbo \
		--max-examples 1000

# =============================================================================
# TRAINING (Seq2Seq Models)
# =============================================================================

train-byt5:
	python -m src.models.seq2seq.trainer configs/seq2seq_byt5.yaml

train-mt5:
	python -m src.models.seq2seq.trainer configs/seq2seq_mt5.yaml

train-mbart:
	python -m src.models.seq2seq.trainer configs/seq2seq_mbart.yaml

# =============================================================================
# INFERENCE (Generate Predictions)
# =============================================================================

infer-byt5:
	python -m src.models.seq2seq.inference \
		--model experiments/byt5-small \
		--test data/revita/splits/test_augmented_random_greedy_errdensity20_clean_seed42.jsonl \
		--output predictions/byt5-small.jsonl \
		--batch-size 16

infer-mt5:
	python -m src.models.seq2seq.inference \
		--model experiments/mt5-base \
		--test data/revita/splits/test_augmented_random_greedy_errdensity20_clean_seed42.jsonl \
		--output predictions/mt5-base.jsonl \
		--batch-size 16

infer-mbart:
	python -m src.models.seq2seq.inference \
		--model experiments/mbart-large \
		--test data/revita/splits/test_augmented_random_greedy_errdensity20_clean_seed42.jsonl \
		--output predictions/mbart-large.jsonl \
		--batch-size 8

# =============================================================================
# EVALUATION
# =============================================================================

eval-byt5:
	python -m src.evaluate \
		--predictions predictions/byt5-small.jsonl \
		--test data/revita/splits/test_augmented_random_greedy_errdensity20_clean_seed42.jsonl \
		--output results/byt5-small.json

eval-mt5:
	python -m src.evaluate \
		--predictions predictions/mt5-base.jsonl \
		--test data/revita/splits/test_augmented_random_greedy_errdensity20_clean_seed42.jsonl \
		--output results/mt5-base.json

eval-mbart:
	python -m src.evaluate \
		--predictions predictions/mbart-large.jsonl \
		--test data/revita/splits/test_augmented_random_greedy_errdensity20_clean_seed42.jsonl \
		--output results/mbart-large.json

error-analysis:
	python -m src.error_analysis --all

# =============================================================================
# REVITA DATA PIPELINE
# =============================================================================

# Complete pipeline: split raw data → augment train/test separately
revita-pipeline:
	python scripts/revita_pipeline_split_and_augment.py \
		--input data/revita/exercise_errors_Finnish_cleaned.jsonl \
		--output-dir data/revita \
		--test-ratio 0.2 \
		--stratify error_count \
		--strategy random \
		--max-error-rate 0.2 \
		--seed 42

# Pipeline with exhaustive augmentation (limited to prevent huge datasets)
revita-pipeline-exhaustive:
	python scripts/revita_pipeline_split_and_augment.py \
		--input data/revita/exercise_errors_Finnish_cleaned.jsonl \
		--output-dir data/revita \
		--test-ratio 0.2 \
		--stratify error_count \
		--strategy exhaustive \
		--max-augmentation 500 \
		--max-error-rate 0.2 \
		--seed 42

# Pipeline for quick testing (limited samples)
revita-pipeline-test:
	python scripts/revita_pipeline_split_and_augment.py \
		--input data/revita/exercise_errors_Finnish_cleaned.jsonl \
		--output-dir data/revita \
		--test-ratio 0.2 \
		--stratify error_count \
		--strategy random \
		--max-error-rate 0.2 \
		--max-augmentation 10 \
		--seed 42

# =============================================================================
# INDIVIDUAL STEPS (for manual control)
# =============================================================================

# Step 1: Split raw data into train/test
revita-split:
	python scripts/revita_split_clean_raw_data.py \
		--input data/revita/exercise_errors_Finnish_cleaned.jsonl \
		--output-dir data/revita/splits \
		--test-ratio 0.2 \
		--stratify error_count \
		--seed 42

# Step 2a: Augment training data
revita-augment-train:
	python scripts/revita_augment_raw_data.py \
		--input data/revita/splits/train_raw.jsonl \
		--output data/revita/splits/train_augmented.jsonl \
		--strategy random \
		--max-error-rate 0.2 \
		--seed 42

# Step 2b: Augment test data
revita-augment-test:
	python scripts/revita_augment_raw_data.py \
		--input data/revita/splits/test_raw.jsonl \
		--output data/revita/splits/test_augmented.jsonl \
		--strategy random \
		--max-error-rate 0.2 \
		--seed 43

# =============================================================================
# EDA REPORTS
# =============================================================================

# Generate EDA for raw cleaned data
revita-eda-raw:
	python scripts/revita_eda_cleaned_raw_data.py \
		--data-file data/revita/exercise_errors_Finnish_cleaned.jsonl \
		--output reports/revita_raw_cleaned_eda.md

# Generate EDA for augmented training data
revita-eda-augmented:
	python scripts/revita_eda_augmented_data.py \
		--input data/revita/splits/train_augmented.jsonl \
		--output reports/revita_train_augmented_eda.md
