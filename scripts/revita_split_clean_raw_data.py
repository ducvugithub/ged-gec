#!/usr/bin/env python3
"""
Split raw Revita data into train/val/test sets BEFORE augmentation.

This ensures that raw examples in training data do not appear in val or test data,
preventing data leakage when augmentation creates multiple samples per raw example.
"""

import json
import argparse
import random
from pathlib import Path
from collections import Counter


def load_jsonl(file_path: Path):
    """Load JSONL file into list of dicts."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def save_jsonl(examples, file_path: Path):
    """Save list of dicts to JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')


def stratified_split(examples, val_ratio=0.1, test_ratio=0.2, stratify_by='error_count', seed=42):
    """
    Perform stratified split based on error count or snippet length.

    Args:
        examples: List of raw examples
        val_ratio: Ratio of validation set (default 0.1 = 10%)
        test_ratio: Ratio of test set (default 0.2 = 20%)
        stratify_by: 'error_count', 'snippet_length', or 'error_density'
        seed: Random seed

    Returns:
        (train_examples, val_examples, test_examples)
    """
    random.seed(seed)

    # Group examples by stratification key
    strata = {}
    for example in examples:
        if stratify_by == 'error_count':
            key = len(example['errors'])
        elif stratify_by == 'snippet_length':
            # Count meaningful words
            words = [t for t in example['snippet'] if t.strip() and t not in [' ', '\n\n']]
            key = len(words) // 5 * 5  # Bucket by 5s: 10-14, 15-19, 20-24, etc.
        elif stratify_by == 'error_density':
            words = [t for t in example['snippet'] if t.strip() and t not in [' ', '\n\n']]
            snippet_length = len(words)
            density = len(example['errors']) / snippet_length if snippet_length > 0 else 0
            key = int(density * 10) * 10  # Bucket by 10%: 0-9%, 10-19%, etc.
        else:
            raise ValueError(f"Unknown stratify_by: {stratify_by}")

        if key not in strata:
            strata[key] = []
        strata[key].append(example)

    # Split each stratum into train/val/test
    train_examples = []
    val_examples = []
    test_examples = []

    for key, stratum in strata.items():
        random.shuffle(stratum)

        # Calculate split indices
        test_idx = int(len(stratum) * (1 - test_ratio))
        val_idx = int(len(stratum) * (1 - test_ratio - val_ratio))

        train_examples.extend(stratum[:val_idx])
        val_examples.extend(stratum[val_idx:test_idx])
        test_examples.extend(stratum[test_idx:])

    # Shuffle final sets
    random.shuffle(train_examples)
    random.shuffle(val_examples)
    random.shuffle(test_examples)

    return train_examples, val_examples, test_examples


def simple_split(examples, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    Simple random split without stratification.

    Args:
        examples: List of raw examples
        val_ratio: Ratio of validation set (default 0.1 = 10%)
        test_ratio: Ratio of test set (default 0.2 = 20%)
        seed: Random seed

    Returns:
        (train_examples, val_examples, test_examples)
    """
    random.seed(seed)
    examples = examples.copy()
    random.shuffle(examples)

    # Calculate split indices
    test_idx = int(len(examples) * (1 - test_ratio))
    val_idx = int(len(examples) * (1 - test_ratio - val_ratio))

    train_examples = examples[:val_idx]
    val_examples = examples[val_idx:test_idx]
    test_examples = examples[test_idx:]

    return train_examples, val_examples, test_examples


def print_split_stats(train_examples, val_examples, test_examples, stratify_by=None):
    """Print statistics about the split."""
    total = len(train_examples) + len(val_examples) + len(test_examples)

    print(f"\n{'='*70}")
    print("SPLIT STATISTICS")
    print(f"{'='*70}")
    print(f"Total examples: {total:,}")
    print(f"Train examples: {len(train_examples):,} ({100 * len(train_examples) / total:.1f}%)")
    print(f"Val examples:   {len(val_examples):,} ({100 * len(val_examples) / total:.1f}%)")
    print(f"Test examples:  {len(test_examples):,} ({100 * len(test_examples) / total:.1f}%)")

    # Error count distribution
    print(f"\n{'='*70}")
    print("ERROR COUNT DISTRIBUTION")
    print(f"{'='*70}")

    train_error_counts = Counter(len(ex['errors']) for ex in train_examples)
    val_error_counts = Counter(len(ex['errors']) for ex in val_examples)
    test_error_counts = Counter(len(ex['errors']) for ex in test_examples)

    all_error_counts = sorted(set(train_error_counts.keys()) | set(val_error_counts.keys()) | set(test_error_counts.keys()))

    print(f"{'Errors':<10} {'Train':<12} {'Val':<12} {'Test':<12} {'Train %':<10} {'Val %':<10} {'Test %':<10}")
    print("-" * 70)
    for count in all_error_counts[:20]:  # Show top 20
        train_count = train_error_counts.get(count, 0)
        val_count = val_error_counts.get(count, 0)
        test_count = test_error_counts.get(count, 0)
        train_pct = 100 * train_count / len(train_examples) if train_examples else 0
        val_pct = 100 * val_count / len(val_examples) if val_examples else 0
        test_pct = 100 * test_count / len(test_examples) if test_examples else 0
        print(f"{count:<10} {train_count:<12,} {val_count:<12,} {test_count:<12,} {train_pct:<10.1f} {val_pct:<10.1f} {test_pct:<10.1f}")

    # Snippet length distribution
    print(f"\n{'='*70}")
    print("SNIPPET LENGTH DISTRIBUTION")
    print(f"{'='*70}")

    def get_snippet_length(ex):
        words = [t for t in ex['snippet'] if t.strip() and t not in [' ', '\n\n']]
        return len(words)

    train_lengths = [get_snippet_length(ex) for ex in train_examples]
    val_lengths = [get_snippet_length(ex) for ex in val_examples]
    test_lengths = [get_snippet_length(ex) for ex in test_examples]

    print(f"Train - Mean: {sum(train_lengths) / len(train_lengths):.1f}, Median: {sorted(train_lengths)[len(train_lengths) // 2]}")
    print(f"Val   - Mean: {sum(val_lengths) / len(val_lengths):.1f}, Median: {sorted(val_lengths)[len(val_lengths) // 2]}")
    print(f"Test  - Mean: {sum(test_lengths) / len(test_lengths):.1f}, Median: {sorted(test_lengths)[len(test_lengths) // 2]}")

    # Error density distribution
    print(f"\n{'='*70}")
    print("ERROR DENSITY DISTRIBUTION")
    print(f"{'='*70}")

    def get_error_density(ex):
        snippet_length = get_snippet_length(ex)
        return len(ex['errors']) / snippet_length if snippet_length > 0 else 0

    train_densities = [get_error_density(ex) for ex in train_examples]
    val_densities = [get_error_density(ex) for ex in val_examples]
    test_densities = [get_error_density(ex) for ex in test_examples]

    print(f"Train - Mean: {100 * sum(train_densities) / len(train_densities):.1f}%")
    print(f"Val   - Mean: {100 * sum(val_densities) / len(val_densities):.1f}%")
    print(f"Test  - Mean: {100 * sum(test_densities) / len(test_densities):.1f}%")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Split raw Revita data into train/val/test sets BEFORE augmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple random split (70% train, 10% val, 20% test)
  python revita_split_clean_raw_data.py

  # Stratified split by error count (80% train, 10% val, 10% test)
  python revita_split_clean_raw_data.py --stratify error_count --val-ratio 0.1 --test-ratio 0.1

  # Stratified split by snippet length
  python revita_split_clean_raw_data.py --stratify snippet_length

  # Stratified split by error density
  python revita_split_clean_raw_data.py --stratify error_density

  # Custom ratios (75/15/10)
  python revita_split_clean_raw_data.py --val-ratio 0.15 --test-ratio 0.10 --stratify error_count
        """
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/revita/exercise_errors_Finnish_cleaned.jsonl'),
        help='Input raw Revita JSONL file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/revita/splits'),
        help='Output directory for train/val/test splits'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1 = 10%%)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.2,
        help='Test set ratio (default: 0.2 = 20%%)'
    )
    parser.add_argument(
        '--stratify',
        type=str,
        choices=['error_count', 'snippet_length', 'error_density', 'none'],
        default='none',
        help='Stratification strategy (default: none = simple random split)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Validate ratios
    if args.val_ratio <= 0 or args.val_ratio >= 1:
        parser.error("val-ratio must be between 0 and 1")
    if args.test_ratio <= 0 or args.test_ratio >= 1:
        parser.error("test-ratio must be between 0 and 1")
    if args.val_ratio + args.test_ratio >= 1:
        parser.error("val-ratio + test-ratio must be less than 1")

    print(f"📂 Loading raw data from {args.input}")
    examples = load_jsonl(args.input)
    print(f"✓ Loaded {len(examples):,} raw examples")

    # Perform split
    train_ratio = 1 - args.val_ratio - args.test_ratio
    print(f"\n🔀 Splitting data...")
    print(f"   Strategy: {args.stratify if args.stratify != 'none' else 'Simple random'}")
    print(f"   Train/Val/Test ratio: {train_ratio*100:.0f}% / {args.val_ratio*100:.0f}% / {args.test_ratio*100:.0f}%")
    print(f"   Random seed: {args.seed}")

    if args.stratify == 'none':
        train_examples, val_examples, test_examples = simple_split(
            examples,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
    else:
        train_examples, val_examples, test_examples = stratified_split(
            examples,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            stratify_by=args.stratify,
            seed=args.seed
        )

    # Print statistics
    print_split_stats(train_examples, val_examples, test_examples, args.stratify)

    # Save splits
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / 'train_raw.jsonl'
    val_path = args.output_dir / 'val_raw.jsonl'
    test_path = args.output_dir / 'test_raw.jsonl'

    save_jsonl(train_examples, train_path)
    save_jsonl(val_examples, val_path)
    save_jsonl(test_examples, test_path)

    print(f"💾 Saved splits:")
    print(f"   Train: {train_path} ({len(train_examples):,} examples)")
    print(f"   Val:   {val_path} ({len(val_examples):,} examples)")
    print(f"   Test:  {test_path} ({len(test_examples):,} examples)")
    print(f"\n✅ Done! Now you can run augmentation separately on each split")


if __name__ == '__main__':
    main()
