"""
Error type distribution analysis — run BEFORE training any models.

Analyzes synthetic and Revita data to understand error type frequencies
and inform test set stratification.
"""

import argparse
from pathlib import Path
import json
from collections import Counter, defaultdict
from typing import Dict, List
import pandas as pd


class ErrorAnalyzer:
    """Analyze error type distributions in GEC datasets."""

    ERROR_TYPE_CATEGORIES = [
        'morphological',
        'syntactic',
        'lexical',
        'agreement',
        'punctuation',
        'other'
    ]

    def __init__(self):
        self.stats = defaultdict(Counter)

    def categorize_error(self, edit: Dict) -> str:
        """
        Categorize an edit into error type.

        Args:
            edit: Edit metadata dict with 'type' key

        Returns:
            Error type category
        """
        # TODO: Implement sophisticated error categorization
        # This should use linguistic analysis, possibly UD features
        edit_type = edit.get('type', 'unknown')

        # Simple heuristic mapping (replace with proper analysis)
        if 'confusion' in edit_type:
            return 'morphological'
        elif 'char' in edit_type:
            return 'other'
        else:
            return 'unknown'

    def analyze_file(self, data_path: Path, source_name: str):
        """
        Analyze error distribution in a single file.

        Args:
            data_path: Path to JSONL data file
            source_name: Name of data source (for reporting)
        """
        error_types = []
        sentence_lengths = []
        edit_counts = []

        with open(data_path) as f:
            for line in f:
                example = json.loads(line)

                sentence_lengths.append(len(example['corrupted'].split()))

                if 'edits' in example:
                    edit_counts.append(len(example['edits']))
                    for edit in example['edits']:
                        error_type = self.categorize_error(edit)
                        error_types.append(error_type)
                        self.stats[source_name][error_type] += 1

        # Summary statistics
        print(f"\n=== {source_name} ===")
        print(f"Total examples: {len(sentence_lengths)}")
        print(f"Avg sentence length: {sum(sentence_lengths) / len(sentence_lengths):.1f} tokens")

        if edit_counts:
            print(f"Avg edits per sentence: {sum(edit_counts) / len(edit_counts):.2f}")

        print("\nError type distribution:")
        total_errors = sum(self.stats[source_name].values())
        for error_type in self.ERROR_TYPE_CATEGORIES:
            count = self.stats[source_name][error_type]
            pct = 100 * count / total_errors if total_errors > 0 else 0
            print(f"  {error_type:20s}: {count:6d} ({pct:5.1f}%)")

    def analyze_all(self, data_paths: Dict[str, Path]):
        """
        Analyze multiple data sources and generate comparative report.

        Args:
            data_paths: Dict mapping source names to file paths
        """
        for source_name, data_path in data_paths.items():
            if data_path.exists():
                self.analyze_file(data_path, source_name)
            else:
                print(f"⚠ Skipping {source_name}: file not found at {data_path}")

        # Generate comparison DataFrame
        rows = []
        for source, counts in self.stats.items():
            total = sum(counts.values())
            row = {'source': source, 'total_errors': total}
            for error_type in self.ERROR_TYPE_CATEGORIES:
                row[error_type] = counts[error_type]
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            print("\n=== Comparison Table ===")
            print(df.to_string(index=False))

            # Save to CSV
            output_path = Path('data/error_distribution_analysis.csv')
            df.to_csv(output_path, index=False)
            print(f"\n✓ Saved analysis to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze error type distributions')
    parser.add_argument('--data', type=Path, help='Single data file to analyze')
    parser.add_argument('--all', action='store_true', help='Analyze all synthetic sources')

    args = parser.parse_args()

    analyzer = ErrorAnalyzer()

    if args.data:
        analyzer.analyze_file(args.data, args.data.stem)
    elif args.all:
        data_paths = {
            'confusion_sets': Path('data/synthetic/confusion_sets/train.jsonl'),
            'random_ops': Path('data/synthetic/random_ops/train.jsonl'),
            'back_translated': Path('data/synthetic/back_translated/train.jsonl'),
            'llm_generated': Path('data/synthetic/llm_generated/train.jsonl'),
            'revita': Path('data/revita/train.jsonl')
        }
        analyzer.analyze_all(data_paths)
    else:
        print("Please specify --data <path> or --all")


if __name__ == '__main__':
    main()
