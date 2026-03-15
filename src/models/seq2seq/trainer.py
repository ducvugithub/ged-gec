"""
Trainer for seq2seq GEC models (T5, mT5, mBART, BART).
"""

from pathlib import Path
from typing import Dict, Any
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from datasets import load_dataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .weighted_trainer import WeightedSeq2SeqTrainer


def load_data(config: Dict[str, Any]):
    """Load and tokenize GEC dataset."""
    data_files = {
        'train': config['data']['train_path'],
        'validation': config['data'].get('val_path'),
        'test': config['data'].get('test_path')
    }

    # Remove None values
    data_files = {k: v for k, v in data_files.items() if v is not None}

    dataset = load_dataset('json', data_files=data_files)
    return dataset


def preprocess_function(examples, tokenizer, config):
    """Tokenize examples for seq2seq training."""
    prefix = config.get('task_prefix', 'grammar: ')

    inputs = [prefix + text for text in examples['corrupted']]
    targets = examples['correct']

    model_inputs = tokenizer(
        inputs,
        max_length=config.get('max_source_length', 512),
        truncation=True
    )

    labels = tokenizer(
        targets,
        max_length=config.get('max_target_length', 512),
        truncation=True
    )

    model_inputs['labels'] = labels['input_ids']

    # Keep training_weight if present
    if 'training_weight' in examples:
        model_inputs['training_weight'] = examples['training_weight']

    return model_inputs


def train(config: Dict[str, Any]):
    """
    Train a seq2seq GEC model.

    Args:
        config: Configuration dictionary
    """
    model_config = config['model']
    training_config = config['training']

    # Initialize wandb
    if training_config.get('use_wandb', False):
        if not WANDB_AVAILABLE:
            print("⚠️  Warning: wandb requested but not installed. Continuing without wandb.")
        else:
            wandb.init(
                project=config.get('project_name', 'finnish-gec'),
                name=config.get('run_name'),
                config=config
            )

    # Load model and tokenizer
    model_name = model_config['pretrained_model']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Load and preprocess data
    dataset = load_data(config)
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, model_config),
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=training_config['output_dir'],
        evaluation_strategy=training_config.get('evaluation_strategy', 'steps'),
        eval_steps=training_config.get('eval_steps', 500),
        learning_rate=float(training_config.get('learning_rate', 5e-5)),
        per_device_train_batch_size=training_config.get('batch_size', 8),
        per_device_eval_batch_size=training_config.get('eval_batch_size', 8),
        num_train_epochs=training_config.get('num_epochs', 3),
        weight_decay=float(training_config.get('weight_decay', 0.01)),
        save_total_limit=training_config.get('save_total_limit', 3),
        predict_with_generate=True,
        fp16=training_config.get('fp16', False),
        push_to_hub=False,
        report_to='wandb' if training_config.get('use_wandb') else 'none'
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Check if we should use weighted training
    use_weights = config.get('data', {}).get('use_weights', False)

    # Initialize trainer (weighted or regular)
    if use_weights:
        print("✅ Using weighted training for balanced learning")
        trainer = WeightedSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset.get('validation'),
            tokenizer=tokenizer,
            data_collator=data_collator,
            use_weights=True
        )
    else:
        print("ℹ️  Using standard training (no weights)")
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset.get('validation'),
            tokenizer=tokenizer,
            data_collator=data_collator
        )

    # Train
    print(f"Starting training: {model_config['type']} ({model_name})")
    trainer.train()

    # Save final model
    final_model_path = Path(training_config['output_dir']) / 'final'
    trainer.save_model(final_model_path)
    print(f"✓ Model saved to {final_model_path}")


if __name__ == '__main__':
    # Example usage
    import yaml
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            config = yaml.safe_load(f)
        train(config)
