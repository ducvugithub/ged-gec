#!/usr/bin/env python3
"""
Exploratory Data Analysis for Augmented Revita GEC Data.

Analyzes the augmented training data to understand:
- Error count distribution
- Snippet length distribution
- Correct frequency distribution (for weight validation)
- Edit distance distribution
- Edit position distribution
- Error rate distribution
- Training weight distribution
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List
import numpy as np
import Levenshtein  # pip install python-Levenshtein


class AugmentedDataEDA:
    """Analyze augmented GEC training data."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.samples = []
        self.stats = defaultdict(list)

    def load_data(self):
        """Load augmented data from JSONL."""
        print(f"📂 Loading data from {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        print(f"✓ Loaded {len(self.samples):,} samples\n")

    def analyze(self):
        """Run all analyses."""
        print("🔍 Analyzing augmented data...\n")

        for sample in self.samples:
            # Basic fields
            num_errors = sample.get('num_errors', 0)
            snippet_length = sample.get('snippet_length', 0)
            error_rate = sample.get('error_rate', 0)
            training_weight = sample.get('training_weight', 1.0)
            correct_frequency = sample.get('correct_frequency', 1)

            corrupted = sample.get('corrupted', '')
            correct = sample.get('correct', '')
            edits = sample.get('edits', [])

            # 1. Error count distribution
            self.stats['error_count'].append(num_errors)

            # 2. Snippet length distribution
            self.stats['snippet_length'].append(snippet_length)

            # 3. Correct frequency distribution
            self.stats['correct_frequency'].append(correct_frequency)

            # 4. Edit distance distribution
            if corrupted and correct:
                edit_dist = Levenshtein.distance(corrupted, correct)
                self.stats['edit_distance'].append(edit_dist)

                # Normalized edit distance (0-1)
                max_len = max(len(corrupted), len(correct))
                if max_len > 0:
                    norm_dist = edit_dist / max_len
                    self.stats['normalized_edit_distance'].append(norm_dist)

            # 5. Edit position distribution (all positions)
            for edit in edits:
                pos = edit.get('position', 0)
                self.stats['edit_positions'].append(pos)

                # Normalized position (0-1 within snippet)
                if snippet_length > 0:
                    norm_pos = pos / snippet_length
                    self.stats['normalized_edit_positions'].append(norm_pos)

            # 6. Error rate distribution
            self.stats['error_rate'].append(error_rate)

            # 7. Training weight distribution
            self.stats['training_weight'].append(training_weight)

            # 8. Text length comparison
            self.stats['corrupted_length'].append(len(corrupted))
            self.stats['correct_length'].append(len(correct))

    def generate_report(self, output_path: Path):
        """Generate markdown report with statistics."""

        report = []

        # Header
        report.append("# Augmented Revita GEC Data - EDA Report")
        report.append("")
        report.append(f"**Dataset:** `{self.data_path.name}`")
        report.append(f"**Total Samples:** {len(self.samples):,}")
        report.append("")
        report.append("---")
        report.append("")

        # 1. Snippet Length Distribution
        report.append("## 1. Snippet Length Distribution")
        report.append("")
        report.append("Length of snippets (in words):")
        report.append("")

        lengths = self.stats['snippet_length']
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        report.append("| Statistic | Value |")
        report.append("|-----------|-------|")
        report.append(f"| Mean | {np.mean(lengths):.1f} words |")
        report.append(f"| Median | {np.median(lengths):.0f} words |")
        report.append(f"| Std Dev | {np.std(lengths):.1f} |")
        report.append(f"| Min | {min(lengths)} words |")
        report.append(f"| Max | {max(lengths)} words |")

        for p in percentiles:
            val = np.percentile(lengths, p)
            report.append(f"| {p}th percentile | {val:.0f} words |")

        report.append("")

        # Length bins
        report.append("**Length Distribution:**")
        report.append("")
        bins = [0, 10, 20, 30, 40, 50, 100, float('inf')]
        bin_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-100', '100+']

        report.append("| Length Range | Count | Percentage |")
        report.append("|--------------|-------|------------|")

        for i in range(len(bins) - 1):
            count = sum(1 for l in lengths if bins[i] < l <= bins[i+1])
            pct = 100 * count / len(lengths)
            report.append(f"| {bin_labels[i]} words | {count:,} | {pct:.2f}% |")

        report.append("")

        # 2. Error Count Distribution
        report.append("## 2. Error Count Distribution")
        report.append("")
        report.append("Distribution of number of errors per sample:")
        report.append("")

        error_counts = Counter(self.stats['error_count'])
        report.append("| Errors | Count | Percentage |")
        report.append("|--------|-------|------------|")

        for num_errors in sorted(error_counts.keys()):
            count = error_counts[num_errors]
            pct = 100 * count / len(self.samples)
            report.append(f"| {num_errors} | {count:,} | {pct:.2f}% |")

        report.append("")
        report.append(f"- **Mean errors per sample:** {np.mean(self.stats['error_count']):.2f}")
        report.append(f"- **Median:** {np.median(self.stats['error_count']):.0f}")
        report.append(f"- **Range:** {min(self.stats['error_count'])} - {max(self.stats['error_count'])}")
        report.append("")

        # 3. Error Rate Distribution
        report.append("## 3. Error Rate Distribution")
        report.append("")
        report.append("Error density (errors / snippet_length):")
        report.append("")

        error_rates = self.stats['error_rate']

        # Convert to percentages for binning
        error_rate_pcts = [r * 100 for r in error_rates]

        report.append("| Error Rate | Count | Percentage |")
        report.append("|------------|-------|------------|")

        rate_bins = [0, 0.001, 5, 10, 15, 20, 100]
        rate_labels = ['0%', '0-5%', '5-10%', '10-15%', '15-20%', '20%+']

        for i in range(len(rate_bins) - 1):
            count = sum(1 for r in error_rate_pcts if rate_bins[i] <= r < rate_bins[i+1])
            pct = 100 * count / len(error_rate_pcts) if error_rate_pcts else 0
            report.append(f"| {rate_labels[i]} | {count:,} | {pct:.2f}% |")

        report.append("")
        report.append(f"- **Mean error rate:** {np.mean(error_rates)*100:.1f}%")
        report.append(f"- **Median error rate:** {np.median(error_rates)*100:.1f}%")
        report.append("")

        # 4. Correct Frequency Distribution
        report.append("## 4. Correct Frequency Distribution")
        report.append("")
        report.append("Shows how many augmented samples were generated from each unique raw example.")
        report.append("Higher frequencies indicate that a raw sample was reused more times during augmentation.")
        report.append("")

        freqs = self.stats['correct_frequency']
        report.append("| Statistic | Value |")
        report.append("|-----------|-------|")
        report.append(f"| Mean frequency | {np.mean(freqs):.1f}× |")
        report.append(f"| Median | {np.median(freqs):.0f}× |")
        report.append(f"| Min | {min(freqs)}× |")
        report.append(f"| Max | {max(freqs)}× |")
        report.append(f"| Std Dev | {np.std(freqs):.1f} |")

        report.append("")

        # Frequency bins
        freq_counter = Counter(freqs)
        report.append("**Top 10 Most Common Frequencies:**")
        report.append("")
        report.append("| Frequency | # of Texts | Total Samples |")
        report.append("|-----------|------------|---------------|")

        for freq, text_count in freq_counter.most_common(10):
            total_samples = freq * text_count
            report.append(f"| {freq}× | {text_count:,} texts | {total_samples:,} samples |")

        report.append("")

        # 5. Edit Distance Distribution
        report.append("## 5. Edit Distance Distribution")
        report.append("")
        report.append("Levenshtein distance between corrupted and correct text:")
        report.append("")

        edit_dists = self.stats['edit_distance']
        norm_dists = self.stats['normalized_edit_distance']

        report.append("| Metric | Absolute | Normalized |")
        report.append("|--------|----------|------------|")
        report.append(f"| Mean | {np.mean(edit_dists):.2f} chars | {np.mean(norm_dists):.3f} |")
        report.append(f"| Median | {np.median(edit_dists):.0f} chars | {np.median(norm_dists):.3f} |")
        report.append(f"| Min | {min(edit_dists)} chars | {min(norm_dists):.3f} |")
        report.append(f"| Max | {max(edit_dists)} chars | {max(norm_dists):.3f} |")
        report.append(f"| Std Dev | {np.std(edit_dists):.2f} | {np.std(norm_dists):.3f} |")

        report.append("")

        # Edit distance bins
        report.append("**Edit Distance Distribution:**")
        report.append("")
        report.append("| Distance | Count | Percentage |")
        report.append("|----------|-------|------------|")

        dist_bins = [0, 1, 2, 5, 10, 20, 50, float('inf')]
        dist_labels = ['0', '1', '2-4', '5-9', '10-19', '20-49', '50+']

        for i in range(len(dist_bins) - 1):
            count = sum(1 for d in edit_dists if dist_bins[i] <= d < dist_bins[i+1])
            pct = 100 * count / len(edit_dists)
            report.append(f"| {dist_labels[i]} | {count:,} | {pct:.2f}% |")

        report.append("")

        # 6. Edit Position Distribution
        report.append("## 6. Edit Position Distribution")
        report.append("")
        report.append("Where in the snippet do errors occur?")
        report.append("")

        edit_positions = self.stats['edit_positions']
        norm_positions = self.stats['normalized_edit_positions']

        report.append("| Statistic | Absolute Position | Relative Position |")
        report.append("|-----------|-------------------|-------------------|")
        report.append(f"| Mean | {np.mean(edit_positions):.1f} | {np.mean(norm_positions):.3f} |")
        report.append(f"| Median | {np.median(edit_positions):.0f} | {np.median(norm_positions):.3f} |")
        report.append(f"| Std Dev | {np.std(edit_positions):.1f} | {np.std(norm_positions):.3f} |")

        report.append("")

        # Position bins (relative)
        report.append("**Position Distribution (relative to snippet length):**")
        report.append("")
        report.append("| Position | Count | Percentage |")
        report.append("|----------|-------|------------|")

        pos_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        pos_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

        for i in range(len(pos_bins) - 1):
            count = sum(1 for p in norm_positions if pos_bins[i] <= p < pos_bins[i+1])
            pct = 100 * count / len(norm_positions) if norm_positions else 0
            report.append(f"| {pos_labels[i]} | {count:,} | {pct:.2f}% |")

        report.append("")

        # 7. Text Length Comparison
        report.append("## 7. Text Length Comparison")
        report.append("")
        report.append("Corrupted vs correct text length (in characters):")
        report.append("")

        corr_lens = self.stats['corrupted_length']
        correct_lens = self.stats['correct_length']

        report.append("| Statistic | Corrupted | Correct | Difference |")
        report.append("|-----------|-----------|---------|------------|")
        report.append(f"| Mean | {np.mean(corr_lens):.1f} | {np.mean(correct_lens):.1f} | {np.mean(corr_lens) - np.mean(correct_lens):.1f} |")
        report.append(f"| Median | {np.median(corr_lens):.0f} | {np.median(correct_lens):.0f} | {np.median(corr_lens) - np.median(correct_lens):.0f} |")

        report.append("")

        # Summary
        report.append("---")
        report.append("")
        report.append("## Summary")
        report.append("")
        report.append(f"- **Total samples:** {len(self.samples):,}")
        report.append(f"- **Unique correct texts:** {len(set(freqs)):,}")
        report.append(f"- **Error count range:** {min(self.stats['error_count'])} - {max(self.stats['error_count'])}")
        report.append(f"- **Average errors per sample:** {np.mean(self.stats['error_count']):.2f}")
        report.append(f"- **Average error rate:** {np.mean(error_rates)*100:.1f}%")
        report.append(f"- **Average edit distance:** {np.mean(edit_dists):.2f} chars")
        report.append(f"- **Average snippet length:** {np.mean(lengths):.1f} words")
        report.append("")

        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"✅ Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='EDA for augmented Revita GEC data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze augmented data
  python scripts/augmented_data_eda.py \\
      --input data/revita/revita_augmented_exhaustive_limit100_errdensity20_clean_seed42.jsonl

  # Analyze train split
  python scripts/augmented_data_eda.py \\
      --input data/revita/train.jsonl \\
      --output reports/train_eda.md
        """
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input augmented data JSONL file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output markdown report (default: reports/augmented_<input>_eda.md)'
    )

    args = parser.parse_args()

    # Auto-generate output path if not provided
    if args.output is None:
        report_name = f"augmented_{args.input.stem}_eda.md"
        args.output = Path('reports') / report_name

    # Create reports directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Run EDA
    eda = AugmentedDataEDA(args.input)
    eda.load_data()
    eda.analyze()
    eda.generate_report(args.output)

    print(f"\n📊 EDA complete!")
    print(f"   Report: {args.output}")


if __name__ == '__main__':
    main()
