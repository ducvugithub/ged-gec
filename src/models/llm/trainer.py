"""
Trainer for LLM-based GEC (LoRA fine-tuning).
"""

from typing import Dict, Any
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset


def train(config: Dict[str, Any]):
    """
    Train (fine-tune) an LLM for GEC using LoRA.

    Args:
        config: Configuration dictionary
    """
    from src.models.llm.model import LLMGECModel, GEC_PROMPT_TEMPLATE

    model_config = config['model']
    training_config = config['training']

    # Initialize model with LoRA
    model = LLMGECModel(
        model_name=model_config['pretrained_model'],
        use_lora=True,
        lora_config=model_config.get('lora', {})
    )

    # Load dataset
    data_files = {'train': config['data']['train_path']}
    if config['data'].get('val_path'):
        data_files['validation'] = config['data']['val_path']

    dataset = load_dataset('json', data_files=data_files)

    # Preprocess: convert to instruction format
    def preprocess_function(examples):
        prompts = []
        for corrupted, correct in zip(examples['corrupted'], examples['correct']):
            prompt = GEC_PROMPT_TEMPLATE.format(input_sentence=corrupted)
            full_text = f"{prompt} {correct}"
            prompts.append(full_text)

        return model.tokenizer(
            prompts,
            truncation=True,
            max_length=model_config.get('max_length', 512)
        )

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config.get('num_epochs', 3),
        per_device_train_batch_size=training_config.get('batch_size', 4),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        learning_rate=training_config.get('learning_rate', 2e-4),
        fp16=training_config.get('fp16', True),
        logging_steps=training_config.get('logging_steps', 10),
        save_strategy="epoch",
        report_to='wandb' if training_config.get('use_wandb') else 'none'
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=model.tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset.get('validation'),
        data_collator=data_collator
    )

    # Train
    print(f"Starting LoRA fine-tuning: {model_config['pretrained_model']}")
    trainer.train()

    # Save LoRA adapters
    model.model.save_pretrained(training_config['output_dir'] + '/lora_adapters')
    print(f"✓ LoRA adapters saved to {training_config['output_dir']}/lora_adapters")


if __name__ == '__main__':
    import yaml
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            config = yaml.safe_load(f)
        train(config)
