"""
Generate predictions from trained seq2seq GEC model.
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import List, Dict
import sys
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    MBartForConditionalGeneration,
    ByT5Tokenizer
)


class Seq2SeqInferencer:
    """Generate predictions from trained seq2seq model."""

    def __init__(self, model_path: Path, batch_size: int = 8):
        """
        Initialize inferencer.

        Args:
            model_path: Path to trained model checkpoint
            batch_size: Batch size for inference
        """
        print(f"Loading model from {model_path}")

        # Load model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.batch_size = batch_size

        print(f"✓ Model loaded on {self.device}")
        print(f"  Model type: {self.model.config.model_type}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def predict_batch(self, corrupted_texts: List[str], max_length: int = 512) -> List[str]:
        """
        Generate predictions for a batch of corrupted texts.

        Args:
            corrupted_texts: List of corrupted sentences
            max_length: Maximum generation length

        Returns:
            List of predicted corrections
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            corrupted_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate predictions
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,  # Beam search for better quality
                early_stopping=True
            )

        # Decode predictions
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return predictions

    def predict_file(self, test_file: Path, output_file: Path, max_length: int = 512):
        """
        Generate predictions for entire test file.

        Args:
            test_file: Input JSONL file with test data
            output_file: Output JSONL file for predictions
            max_length: Maximum sequence length
        """
        # Load test data
        print(f"\nLoading test data from {test_file}")
        with open(test_file, encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f if line.strip()]

        print(f"Loaded {len(test_data):,} test examples")

        # Extract corrupted texts and references
        corrupted_texts = [example['corrupted'] for example in test_data]
        references = [example['correct'] for example in test_data]

        # Generate predictions in batches
        print(f"\nGenerating predictions (batch_size={self.batch_size})...")
        all_predictions = []

        for i in tqdm(range(0, len(corrupted_texts), self.batch_size)):
            batch_corrupted = corrupted_texts[i:i + self.batch_size]
            batch_predictions = self.predict_batch(batch_corrupted, max_length)
            all_predictions.extend(batch_predictions)

        # Create output records
        print(f"\nSaving predictions to {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for corrupted, reference, prediction in zip(corrupted_texts, references, all_predictions):
                record = {
                    'corrupted': corrupted,
                    'reference': reference,
                    'prediction': prediction
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"✓ Saved {len(all_predictions):,} predictions")

        # Show a few examples
        print("\n📝 Sample Predictions:")
        print("-" * 80)
        for i in range(min(3, len(all_predictions))):
            print(f"\nExample {i+1}:")
            print(f"  Corrupted:  {corrupted_texts[i]}")
            print(f"  Reference:  {references[i]}")
            print(f"  Predicted:  {all_predictions[i]}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Generate predictions from trained seq2seq GEC model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate predictions from trained model
  python src/models/seq2seq/inference.py \\
      --model /scratch/project_2006601/GEC/experiments/byt5-small \\
      --test data/revita/splits/test_augmented_random_greedy_errdensity20_clean_seed42.jsonl \\
      --output predictions/byt5-small.jsonl

  # Adjust batch size for GPU memory
  python src/models/seq2seq/inference.py \\
      --model experiments/byt5-small \\
      --test data/revita/splits/test_*.jsonl \\
      --output predictions/byt5-small.jsonl \\
      --batch-size 16
        """
    )
    parser.add_argument('--model', type=Path, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test', type=Path, required=True,
                       help='Test data JSONL file')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output JSONL file for predictions')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for inference (default: 8)')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length (default: 512)')

    args = parser.parse_args()

    # Verify files exist
    if not args.model.exists():
        print(f"Error: Model checkpoint not found: {args.model}")
        sys.exit(1)

    if not args.test.exists():
        print(f"Error: Test file not found: {args.test}")
        sys.exit(1)

    # Run inference
    inferencer = Seq2SeqInferencer(args.model, args.batch_size)
    inferencer.predict_file(args.test, args.output, args.max_length)


if __name__ == '__main__':
    main()
