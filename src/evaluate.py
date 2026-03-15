"""
GEC Evaluation: F0.5, GLEU, Exact Match, and stratified analysis.
"""

import argparse
from pathlib import Path
from typing import Dict, List
import yaml
import json
from collections import defaultdict
import sys

# Import metrics from metrics module
from metrics import (
    compute_all_metrics,
    compute_f05_simple,
    compute_gleu,
    exact_match_accuracy
)


class GECEvaluator:
    """Evaluate GEC model outputs with multiple metrics."""

    def __init__(self, use_errant: bool = False):
        """
        Initialize evaluator.

        Args:
            use_errant: Whether to use ERRANT for F0.5 (requires errant package)
        """
        self.use_errant = use_errant

        if use_errant:
            try:
                from metrics import compute_f05_errant
                self.compute_f05 = compute_f05_errant
                print("✓ Using ERRANT for F0.5 computation")
            except ImportError:
                print("Warning: ERRANT not available, falling back to simple F0.5")
                self.use_errant = False

    def evaluate(self,
                predictions_file: Path,
                test_data_file: Path) -> Dict:
        """
        Run full evaluation pipeline.

        Args:
            predictions_file: JSONL file with model predictions
            test_data_file: JSONL file with test data

        Returns:
            Dictionary of evaluation metrics
        """
        # Load data
        print(f"Loading predictions from {predictions_file}")
        with open(predictions_file, encoding='utf-8') as f:
            pred_data = [json.loads(line) for line in f if line.strip()]

        predictions = [p['prediction'] for p in pred_data]
        references = [p['reference'] for p in pred_data]
        sources = [p['corrupted'] for p in pred_data]

        print(f"Loaded {len(predictions):,} predictions")

        # Compute all metrics
        print("\nComputing metrics...")
        metrics = compute_all_metrics(predictions, references, sources)

        # Stratified metrics
        print("Computing stratified metrics...")
        stratified = self._stratify_by_error_count(pred_data, predictions, references, sources)

        return {
            'aggregate': metrics,
            'by_error_count': stratified,
            'num_examples': len(predictions)
        }

    def _stratify_by_error_count(self,
                                  examples: List[Dict],
                                  predictions: List[str],
                                  references: List[str],
                                  sources: List[str]) -> Dict[str, Dict]:
        """
        Compute metrics stratified by error count.

        Args:
            examples: List of examples with metadata
            predictions: Model predictions
            references: Gold corrections
            sources: Original corrupted sentences

        Returns:
            Dictionary mapping error counts to metric dicts
        """
        error_count_groups = defaultdict(lambda: {'predictions': [], 'references': [], 'sources': []})

        for example, pred, ref, src in zip(examples, predictions, references, sources):
            error_count = example.get('num_errors', 0)

            # Group by error count ranges
            if error_count == 0:
                key = '0_errors'
            elif error_count == 1:
                key = '1_error'
            elif error_count <= 3:
                key = '2-3_errors'
            elif error_count <= 5:
                key = '4-5_errors'
            else:
                key = '6+_errors'

            error_count_groups[key]['predictions'].append(pred)
            error_count_groups[key]['references'].append(ref)
            error_count_groups[key]['sources'].append(src)

        results = {}
        for key, group in error_count_groups.items():
            preds = group['predictions']
            refs = group['references']
            srcs = group['sources']

            if len(preds) > 0:
                group_metrics = compute_all_metrics(preds, refs, srcs)
                group_metrics['count'] = len(preds)
                results[key] = group_metrics

        return results


def print_results(results: Dict):
    """Pretty print evaluation results."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    # Aggregate metrics
    print("\n📊 Aggregate Metrics")
    print("-" * 70)

    agg = results['aggregate']
    print(f"Total examples: {results['num_examples']:,}")
    print(f"\nPrimary Metrics:")
    print(f"  F0.5 (precision-weighted): {agg.get('f05', 0):.2f}%")
    print(f"  GLEU (GEC-specific):       {agg.get('gleu', 0):.2f}")
    print(f"  Exact Match:               {agg.get('exact_match', 0):.2f}%")

    print(f"\nDetailed F0.5 Breakdown:")
    print(f"  Precision: {agg.get('precision', 0):.2f}%")
    print(f"  Recall:    {agg.get('recall', 0):.2f}%")

    print(f"\nEdit Distance:")
    print(f"  Avg Character Edit Distance: {agg.get('avg_char_edit_distance', 0):.2f}")
    print(f"  Avg Word Edit Distance:      {agg.get('avg_word_edit_distance', 0):.2f}")

    # Stratified metrics
    if 'by_error_count' in results and results['by_error_count']:
        print("\n📈 Stratified by Error Count")
        print("-" * 70)
        print(f"{'Group':<15} {'Count':<10} {'F0.5':<10} {'GLEU':<10} {'Exact Match':<12}")
        print("-" * 70)

        # Sort by error count
        error_order = ['0_errors', '1_error', '2-3_errors', '4-5_errors', '6+_errors']
        for key in error_order:
            if key in results['by_error_count']:
                metrics = results['by_error_count'][key]
                print(f"{key:<15} {metrics['count']:<10,} "
                      f"{metrics.get('f05', 0):<10.2f} "
                      f"{metrics.get('gleu', 0):<10.2f} "
                      f"{metrics.get('exact_match', 0):<12.2f}")

    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GEC model predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate predictions file
  python src/evaluate.py \\
      --predictions predictions/byt5-small.jsonl \\
      --test data/revita/splits/test_augmented_*.jsonl

  # Save results to JSON
  python src/evaluate.py \\
      --predictions predictions/byt5-small.jsonl \\
      --test data/revita/splits/test_augmented_*.jsonl \\
      --output results/byt5-small.json

  # Use ERRANT for F0.5 (if installed)
  python src/evaluate.py \\
      --predictions predictions/byt5-small.jsonl \\
      --test data/revita/splits/test_augmented_*.jsonl \\
      --use-errant
        """
    )
    parser.add_argument('--predictions', type=Path, required=True,
                       help='Predictions JSONL file')
    parser.add_argument('--test', type=Path, required=True,
                       help='Test data JSONL file')
    parser.add_argument('--output', type=Path,
                       help='Output JSON file for results (optional)')
    parser.add_argument('--use-errant', action='store_true',
                       help='Use ERRANT for F0.5 computation (requires errant package)')

    args = parser.parse_args()

    # Verify files exist
    if not args.predictions.exists():
        print(f"Error: Predictions file not found: {args.predictions}")
        sys.exit(1)

    if not args.test.exists():
        print(f"Error: Test file not found: {args.test}")
        sys.exit(1)

    # Run evaluation
    evaluator = GECEvaluator(use_errant=args.use_errant)
    results = evaluator.evaluate(args.predictions, args.test)

    # Print results
    print_results(results)

    # Save results if output specified
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to {args.output}")


if __name__ == '__main__':
    main()
