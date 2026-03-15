"""
Generate predictions from a trained GEC model.

Usage:
    python src/generate_predictions.py \
        --model experiments/byt5-small/best \
        --test data/revita/splits/test_augmented_*.jsonl \
        --output predictions/byt5-small.jsonl
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_test_data(test_path: Path):
    """Load test data from JSONL file."""
    examples = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def generate_predictions(model, tokenizer, examples, batch_size=8, max_length=512, device='cuda'):
    """
    Generate predictions for all examples.

    Args:
        model: Loaded seq2seq model
        tokenizer: Loaded tokenizer
        examples: List of test examples
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        device: Device to run on

    Returns:
        List of predictions
    """
    model.eval()
    model.to(device)

    predictions = []

    # Process in batches
    for i in tqdm(range(0, len(examples), batch_size), desc="Generating predictions"):
        batch = examples[i:i + batch_size]

        # Extract corrupted sentences
        corrupted_texts = [ex['corrupted'] for ex in batch]

        # Tokenize
        inputs = tokenizer(
            corrupted_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        # Decode
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_predictions)

    return predictions


def main():
    parser = argparse.ArgumentParser(description='Generate predictions from trained GEC model')
    parser.add_argument('--model', type=Path, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test', type=Path, required=True,
                       help='Path to test data JSONL file')
    parser.add_argument('--output', type=Path, required=True,
                       help='Path to save predictions JSONL file')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda/cpu)')

    args = parser.parse_args()

    print(f"Loading model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    print(f"Loading test data from {args.test}")
    examples = load_test_data(args.test)
    print(f"Loaded {len(examples):,} examples")

    print(f"Generating predictions...")
    predictions = generate_predictions(
        model,
        tokenizer,
        examples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )

    print(f"Saving predictions to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        for ex, pred in zip(examples, predictions):
            result = {
                'corrupted': ex['corrupted'],
                'reference': ex['correct'],
                'prediction': pred,
                'num_errors': ex.get('num_errors', 0)
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"✅ Saved {len(predictions):,} predictions to {args.output}")


if __name__ == '__main__':
    main()
