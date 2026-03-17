#!/usr/bin/env python3
"""
Generate combined train/val/test statistics report.

Combines key stats from all splits into one concise report for easy comparison.
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np


def analyze_split(file_path: Path):
    """Analyze a single split file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Extract statistics
    corrupted_lengths = []
    correct_lengths = []
    num_errors_list = []
    error_rates = []

    for sample in data:
        corrupted = sample.get('corrupted', '')
        correct = sample.get('correct', '')

        corrupted_lengths.append(len(corrupted.split()))
        correct_lengths.append(len(correct.split()))
        num_errors_list.append(sample.get('num_errors', 0))
        error_rates.append(sample.get('error_rate', 0))

    return {
        'total': len(data),
        'corrupted_lengths': corrupted_lengths,
        'correct_lengths': correct_lengths,
        'num_errors': num_errors_list,
        'error_rates': error_rates
    }


def format_split_section(split_name: str, stats: dict) -> list:
    """Format a section for one split."""
    lines = []

    lines.append("=" * 70)
    lines.append(f"{split_name.upper()} DATA")
    lines.append("=" * 70)
    lines.append("")

    # Dataset size
    lines.append(f"**Total samples:** {stats['total']:,}")
    lines.append("")

    # Sequence lengths
    corr_lens = stats['corrupted_lengths']
    correct_lens = stats['correct_lengths']

    lines.append("Corrupted sequences:")
    lines.append(f"  Mean: {np.mean(corr_lens):.1f} words")
    lines.append(f"  Median: {np.median(corr_lens):.0f} words")
    lines.append(f"  Max: {max(corr_lens)} words")
    lines.append(f"  95th percentile: {np.percentile(corr_lens, 95):.0f} words")
    lines.append(f"  99th percentile: {np.percentile(corr_lens, 99):.0f} words")
    lines.append("")

    lines.append("Correct sequences:")
    lines.append(f"  Mean: {np.mean(correct_lens):.1f} words")
    lines.append(f"  Median: {np.median(correct_lens):.0f} words")
    lines.append(f"  Max: {max(correct_lens)} words")
    lines.append(f"  95th percentile: {np.percentile(correct_lens, 95):.0f} words")
    lines.append(f"  99th percentile: {np.percentile(correct_lens, 99):.0f} words")
    lines.append("")

    # Token estimates
    lines.append("Estimated subword tokens (assuming 2x multiplier for Finnish):")
    lines.append(f"  Max corrupted: ~{max(corr_lens) * 2:.0f} tokens")
    lines.append(f"  Max correct: ~{max(correct_lens) * 2:.0f} tokens")
    lines.append(f"  95th percentile corrupted: ~{np.percentile(corr_lens, 95) * 2:.0f} tokens")
    lines.append(f"  99th percentile corrupted: ~{np.percentile(corr_lens, 99) * 2:.0f} tokens")
    lines.append("")

    # Error count distribution
    error_counter = Counter(stats['num_errors'])
    lines.append("**Error Count Distribution:**")
    lines.append("")

    # Show top error counts (limit to avoid clutter)
    for num_errors in sorted(error_counter.keys())[:8]:  # Show first 8
        count = error_counter[num_errors]
        pct = 100 * count / stats['total']
        lines.append(f"  {num_errors} errors: {count:,} ({pct:.1f}%)")

    if len(error_counter) > 8:
        lines.append(f"  ... and {len(error_counter) - 8} more error counts")

    lines.append("")
    lines.append(f"  Mean: {np.mean(stats['num_errors']):.2f} errors/sample")
    lines.append(f"  Median: {np.median(stats['num_errors']):.0f} errors/sample")
    lines.append("")

    # Error density distribution
    error_rates = [r * 100 for r in stats['error_rates']]

    lines.append("**Error Density Distribution:**")
    lines.append("")

    bins = [
        (0, 0.001, '0% (no errors)'),
        (0.001, 5, '0-5%'),
        (5, 10, '5-10%'),
        (10, 15, '10-15%'),
        (15, 20, '15-20%'),
        (20, 100, '20%+')
    ]

    for low, high, label in bins:
        count = sum(1 for r in error_rates if low <= r < high)
        pct = 100 * count / stats['total']
        lines.append(f"  {label:<20} {count:>8,} ({pct:>5.1f}%)")

    lines.append("")
    lines.append(f"  Mean: {np.mean(error_rates):.2f}%")
    lines.append(f"  Median: {np.median(error_rates):.2f}%")
    lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser(
        description='Generate combined train/val/test statistics report'
    )

    parser.add_argument(
        '--train',
        type=Path,
        default=Path('data/revita/splits/train_augmented_random_greedy_errdensity20_clean_seed42.jsonl'),
        help='Training split JSONL'
    )
    parser.add_argument(
        '--val',
        type=Path,
        default=Path('data/revita/splits/val_augmented_random_greedy_errdensity20_clean_seed42.jsonl'),
        help='Validation split JSONL'
    )
    parser.add_argument(
        '--test',
        type=Path,
        default=Path('data/revita/splits/test_augmented_random_greedy_errdensity20_clean_seed42.jsonl'),
        help='Test split JSONL'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('reports/train_val_test_stats.md'),
        help='Output markdown report'
    )

    args = parser.parse_args()

    # Verify files exist
    for name, path in [('train', args.train), ('val', args.val), ('test', args.test)]:
        if not path.exists():
            print(f"❌ Error: {name} file not found: {path}")
            return

    # Analyze splits
    print("🔍 Analyzing splits...")
    train_stats = analyze_split(args.train)
    val_stats = analyze_split(args.val)
    test_stats = analyze_split(args.test)
    print(f"  Train: {train_stats['total']:,} samples")
    print(f"  Val:   {val_stats['total']:,} samples")
    print(f"  Test:  {test_stats['total']:,} samples")

    # Generate report
    print("\n📝 Generating combined report...")
    report = []

    # Header
    report.append("# Train/Val/Test Split Statistics")
    report.append("")
    report.append("Combined statistics for all data splits.")
    report.append("")

    # Individual splits
    report.extend(format_split_section("Training", train_stats))
    report.extend(format_split_section("Validation", val_stats))
    report.extend(format_split_section("Test", test_stats))

    report.append("=" * 70)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"✅ Report saved to: {args.output}")


if __name__ == '__main__':
    main()
