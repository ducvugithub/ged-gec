#!/usr/bin/env python3
"""
Enhanced EDA for Revita learner error data.

Features:
1. Meaningful word counting (excludes punctuation)
2. Meaningful word frequency ranking
3. Error count distribution
4. Error instance count distribution
5. Error density distribution
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any
import re
from datetime import datetime
from math import comb


class RevitaEDA:
    """Enhanced EDA for Revita-specific format."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.examples = []
        self.stats = {}

    def load_data(self) -> int:
        """Load Revita JSONL data."""
        print(f"📂 Loading data from {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        print(f"✓ Loaded {len(self.examples)} examples")
        return len(self.examples)

    def _is_meaningful_word(self, token: str) -> bool:
        """Check if token is a meaningful word (not punctuation or whitespace)."""
        if not token or not token.strip():
            return False
        # Exclude pure punctuation
        if all(not c.isalnum() for c in token):
            return False
        # Exclude common separators
        if token in [' ', '\n', '\n\n', '\t']:
            return False
        return True

    def _get_meaningful_words(self, snippet: List[str]) -> List[str]:
        """Extract meaningful words from snippet."""
        return [t for t in snippet if self._is_meaningful_word(t)]

    def analyze_all(self) -> Dict:
        """Run complete analysis."""
        print("\n📊 Running comprehensive analysis...\n")

        self.stats = {
            'basic': self._analyze_basic(),
            'errors': self._analyze_errors(),
            'linguistic': self._analyze_linguistic(),
            'error_patterns': self._analyze_error_patterns(),
            'distributions': self._analyze_distributions(),
            'exhaustive_potential': self._analyze_exhaustive_potential()
        }

        return self.stats

    def _analyze_basic(self) -> Dict:
        """Basic statistics with meaningful word counting."""
        print("  → Basic statistics...")

        snippet_lengths = []  # Meaningful words only

        for ex in self.examples:
            snippet = ex.get('snippet', [])
            meaningful_words = self._get_meaningful_words(snippet)
            snippet_lengths.append(len(meaningful_words))

        return {
            'total_examples': len(self.examples),
            'avg_snippet_length': sum(snippet_lengths) / len(snippet_lengths) if snippet_lengths else 0,
            'min_length': min(snippet_lengths) if snippet_lengths else 0,
            'max_length': max(snippet_lengths) if snippet_lengths else 0,
            'median_length': sorted(snippet_lengths)[len(snippet_lengths)//2] if snippet_lengths else 0
        }

    def _analyze_errors(self) -> Dict:
        """Analyze error instances."""
        print("  → Error analysis...")

        total_errors = 0
        total_instances = 0
        errors_per_example = []
        instances_per_error = []
        error_lengths = Counter()  # Single word vs multi-word errors

        for ex in self.examples:
            errors = ex.get('errors', [])
            errors_per_example.append(len(errors))
            total_errors += len(errors)

            for error in errors:
                instances = error.get('instances', [])
                total_instances += len(instances)
                instances_per_error.append(len(instances))

                # Check if multi-word error (by wid length)
                wid = error.get('wid', [])
                word_count = len(wid)
                error_lengths[word_count] += 1

        return {
            'total_errors': total_errors,
            'total_instances': total_instances,
            'avg_errors_per_example': total_errors / len(self.examples) if self.examples else 0,
            'avg_instances_per_error': total_instances / total_errors if total_errors else 0,
            'min_errors': min(errors_per_example) if errors_per_example else 0,
            'max_errors': max(errors_per_example) if errors_per_example else 0,
            'error_length_dist': dict(error_lengths),
            'errors_per_example_list': errors_per_example,
            'instances_per_error_list': instances_per_error
        }

    def _analyze_linguistic(self) -> Dict:
        """Analyze linguistic patterns with meaningful words only."""
        print("  → Linguistic patterns...")

        correct_words = []
        incorrect_instances = []
        word_frequencies = Counter()

        for ex in self.examples:
            # Collect meaningful words only
            snippet = ex.get('snippet', [])
            meaningful_words = self._get_meaningful_words(snippet)
            # Lowercase for counting
            word_frequencies.update([w.lower() for w in meaningful_words])

            # Collect error information
            for error in ex.get('errors', []):
                correct_words.append(error.get('word', ''))
                incorrect_instances.extend(error.get('instances', []))

        # Analyze word lengths
        avg_word_length = sum(len(w) for w in word_frequencies.keys()) / len(word_frequencies) if word_frequencies else 0

        return {
            'unique_words': len(word_frequencies),
            'total_word_occurrences': sum(word_frequencies.values()),
            'avg_word_length': avg_word_length,
            'top_20_words': word_frequencies.most_common(20),
            'unique_correct_forms': len(set(correct_words)),
            'unique_incorrect_forms': len(set(incorrect_instances))
        }

    def _analyze_distributions(self) -> Dict:
        """Analyze error count, instance count, and error density distributions."""
        print("  → Distribution analysis...")

        error_counts = Counter()
        instance_counts = Counter()
        density_bins = Counter()

        for ex in self.examples:
            snippet = ex.get('snippet', [])
            errors = ex.get('errors', [])

            # Meaningful words count
            meaningful_words = self._get_meaningful_words(snippet)
            snippet_length = len(meaningful_words)

            # Error count distribution
            error_count = len(errors)
            error_counts[error_count] += 1

            # Instance count distribution
            for error in errors:
                instances = error.get('instances', [])
                instance_counts[len(instances)] += 1

            # Error density distribution
            if snippet_length > 0:
                density = error_count / snippet_length
                # Bin into 5% intervals
                density_bin = int(density * 20) * 5  # 0%, 5%, 10%, 15%, ...
                density_bins[density_bin] += 1

        return {
            'error_count_dist': dict(sorted(error_counts.items())),
            'instance_count_dist': dict(sorted(instance_counts.items())),
            'error_density_dist': dict(sorted(density_bins.items()))
        }

    def _analyze_exhaustive_potential(self) -> Dict:
        """Estimate potential exhaustive sampling counts (NO LIMIT - truly exhaustive)."""
        print("  → Exhaustive sampling potential analysis (unlimited)...")

        total_potential = 0
        potential_per_example = []
        snippet_length_dist = Counter()
        potential_by_error_count = defaultdict(int)
        examples_by_potential_range = Counter()

        for ex in self.examples:
            snippet = ex.get('snippet', [])
            errors = ex.get('errors', [])

            # Get meaningful word count
            meaningful_words = self._get_meaningful_words(snippet)
            snippet_length = len(meaningful_words)
            snippet_length_dist[snippet_length] += 1

            # NO LIMIT - generate for ALL error counts from 1 to total_errors
            total_errors = len(errors)

            # Calculate average instances per error for this example
            total_instances = 0
            for error in errors:
                total_instances += len(error.get('instances', []))
            avg_instances = total_instances / total_errors if total_errors else 0

            # Calculate truly exhaustive combinations (1 to total_errors)
            example_potential = 1  # Clean sample

            for k in range(1, total_errors + 1):
                # C(n, k) × avg_instances^k
                combos = comb(total_errors, k) * (avg_instances ** k)
                example_potential += combos
                potential_by_error_count[k] += combos

            total_potential += example_potential
            potential_per_example.append(example_potential)

            # Categorize by potential range
            if example_potential < 100:
                examples_by_potential_range['<100'] += 1
            elif example_potential < 500:
                examples_by_potential_range['100-500'] += 1
            elif example_potential < 1000:
                examples_by_potential_range['500-1K'] += 1
            elif example_potential < 5000:
                examples_by_potential_range['1K-5K'] += 1
            elif example_potential < 10000:
                examples_by_potential_range['5K-10K'] += 1
            elif example_potential < 50000:
                examples_by_potential_range['10K-50K'] += 1
            elif example_potential < 100000:
                examples_by_potential_range['50K-100K'] += 1
            elif example_potential < 1000000:
                examples_by_potential_range['100K-1M'] += 1
            else:
                examples_by_potential_range['>1M'] += 1

        # Calculate statistics
        avg_potential = sum(potential_per_example) / len(potential_per_example) if potential_per_example else 0
        median_potential = sorted(potential_per_example)[len(potential_per_example)//2] if potential_per_example else 0
        max_potential = max(potential_per_example) if potential_per_example else 0
        min_potential = min(potential_per_example) if potential_per_example else 0

        return {
            'total_potential_samples': int(total_potential),
            'avg_potential_per_example': avg_potential,
            'median_potential_per_example': median_potential,
            'min_potential': min_potential,
            'max_potential': max_potential,
            'potential_by_error_count': dict(sorted(potential_by_error_count.items())[:15]),  # Top 15
            'examples_by_potential_range': dict(sorted(examples_by_potential_range.items(),
                                                       key=lambda x: ['<100', '100-500', '500-1K', '1K-5K', '5K-10K',
                                                                     '10K-50K', '50K-100K', '100K-1M', '>1M'].index(x[0]))),
            'snippet_length_dist': dict(sorted(snippet_length_dist.items())[:30])  # Top 30 lengths
        }

    def _analyze_error_patterns(self) -> Dict:
        """Analyze common error patterns."""
        print("  → Error pattern analysis...")

        error_categories = {
            'case_errors': 0,
            'tense_errors': 0,
            'spelling_errors': 0,
            'compound_errors': 0,
            'english_interference': 0,
            'other': 0
        }

        case_suffixes = ['ssa', 'ssä', 'sta', 'stä', 'lle', 'lta', 'ltä', 'lla', 'llä', 'ksi', 'na', 'nä', 'n', 'a', 'ä']

        for ex in self.examples:
            for error in ex.get('errors', []):
                correct = error.get('word', '').lower()
                instances = error.get('instances', [])

                for instance in instances:
                    instance_lower = instance.lower()

                    # English interference
                    if any(c.isascii() and not c.isalpha() for c in instance) or \
                       instance in ['do', 'is doing', 'to do', 'She', 'center']:
                        error_categories['english_interference'] += 1

                    # Compound word errors
                    elif ' ' in correct and ' ' not in instance:
                        error_categories['compound_errors'] += 1

                    # Case errors
                    elif any(correct.endswith(suffix) and instance_lower.startswith(correct[:-len(suffix)])
                            for suffix in case_suffixes):
                        error_categories['case_errors'] += 1

                    # Spelling errors
                    elif self._edit_distance(correct, instance_lower) <= 2:
                        error_categories['spelling_errors'] += 1

                    else:
                        error_categories['other'] += 1

        return error_categories

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Simple edit distance calculation."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def generate_report(self, output_path: Path):
        """Generate comprehensive markdown report."""
        print(f"\n📝 Generating report...")

        lines = []

        # Header
        lines.append("# Revita Finnish Learner Error Data - EDA Report")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Data Source:** `{self.data_path}`")
        lines.append("\n---\n")

        # Basic Statistics
        basic = self.stats['basic']
        lines.append("## 1. Dataset Overview\n")
        lines.append(f"- **Total Examples:** {basic['total_examples']:,}")
        lines.append(f"- **Average Snippet Length:** {basic['avg_snippet_length']:.1f} words (meaningful words only)")
        lines.append(f"- **Median Snippet Length:** {basic['median_length']} words")
        lines.append(f"- **Snippet Length Range:** {basic['min_length']} - {basic['max_length']} words")

        # Error Statistics
        errors = self.stats['errors']
        lines.append("\n---\n")
        lines.append("## 2. Error Statistics\n")
        lines.append(f"- **Total Error Targets:** {errors['total_errors']:,}")
        lines.append(f"- **Total Error Instances:** {errors['total_instances']:,}")
        lines.append(f"- **Average Errors per Example:** {errors['avg_errors_per_example']:.2f}")
        lines.append(f"- **Average Instances per Error:** {errors['avg_instances_per_error']:.2f}")
        lines.append(f"- **Error Range per Example:** {errors['min_errors']} - {errors['max_errors']}")

        lines.append("\n### Error Complexity (by word count)\n")
        lines.append("| Type | Count | Percentage |")
        lines.append("|------|-------|------------|")
        total_errors = sum(errors['error_length_dist'].values())
        for word_count in sorted(errors['error_length_dist'].keys()):
            count = errors['error_length_dist'][word_count]
            pct = 100 * count / total_errors if total_errors else 0
            lines.append(f"| {word_count} Word Error{'s' if word_count > 1 else ''} | {count:,} | {pct:.1f}% |")

        # Linguistic Patterns
        ling = self.stats['linguistic']
        lines.append("\n---\n")
        lines.append("## 3. Linguistic Analysis\n")
        lines.append(f"- **Unique Meaningful Words:** {ling['unique_words']:,}")
        lines.append(f"- **Total Word Occurrences:** {ling['total_word_occurrences']:,}")
        lines.append(f"- **Average Word Length:** {ling['avg_word_length']:.2f} characters")
        lines.append(f"- **Unique Correct Forms:** {ling['unique_correct_forms']:,}")
        lines.append(f"- **Unique Error Forms:** {ling['unique_incorrect_forms']:,}")

        lines.append("\n### Top 20 Most Frequent Meaningful Words\n")
        lines.append("| Rank | Word | Frequency |")
        lines.append("|------|------|-----------| ")
        for rank, (word, freq) in enumerate(ling['top_20_words'], 1):
            lines.append(f"| {rank} | {word} | {freq:,} |")

        # Distributions
        dist = self.stats['distributions']

        lines.append("\n---\n")
        lines.append("## 4. Error Count Distribution\n")
        lines.append("**How many errors each example has:**\n")
        lines.append("| Error Count | Examples | Percentage |")
        lines.append("|-------------|----------|------------|")
        total_examples = basic['total_examples']
        for error_count, example_count in sorted(dist['error_count_dist'].items())[:20]:  # Top 20
            pct = 100 * example_count / total_examples if total_examples else 0
            lines.append(f"| {error_count} | {example_count:,} | {pct:.1f}% |")

        lines.append("\n---\n")
        lines.append("## 5. Instance Count Distribution\n")
        lines.append("**How many instances each error has:**\n")
        lines.append("| Instances | Error Count | Percentage |")
        lines.append("|-----------|-------------|------------|")
        total_error_targets = errors['total_errors']
        for instance_count, error_count in sorted(dist['instance_count_dist'].items())[:15]:
            pct = 100 * error_count / total_error_targets if total_error_targets else 0
            lines.append(f"| {instance_count} | {error_count:,} | {pct:.1f}% |")

        lines.append("\n---\n")
        lines.append("## 6. Error Density Distribution\n")
        lines.append("**Error density = (error count / snippet length):**\n")
        lines.append("| Density Range | Examples | Percentage |")
        lines.append("|---------------|----------|------------|")
        for density_bin, example_count in sorted(dist['error_density_dist'].items())[:20]:
            density_pct = density_bin
            next_bin = density_bin + 5
            pct = 100 * example_count / total_examples if total_examples else 0
            lines.append(f"| {density_pct}%-{next_bin}% | {example_count:,} | {pct:.1f}% |")

        # Exhaustive Potential
        exhaust = self.stats['exhaustive_potential']
        lines.append("\n---\n")
        lines.append("## 7. Exhaustive Sampling Potential (NO LIMIT)\n")
        lines.append(f"**If we generate ALL possible error combinations:**\n")
        lines.append(f"- **Total Potential Samples:** {exhaust['total_potential_samples']:,} 🤯")
        lines.append(f"- **Average per Example:** {exhaust['avg_potential_per_example']:,.0f}")
        lines.append(f"- **Median per Example:** {exhaust['median_potential_per_example']:,.0f}")
        lines.append(f"- **Range:** {exhaust['min_potential']:,.0f} - {exhaust['max_potential']:,.0f}")

        lines.append("\n### Examples by Potential Sample Count\n")
        lines.append("| Potential Range | Examples | Percentage |")
        lines.append("|-----------------|----------|------------|")
        for range_label, example_count in exhaust['examples_by_potential_range'].items():
            pct = 100 * example_count / total_examples if total_examples else 0
            lines.append(f"| {range_label} samples | {example_count:,} | {pct:.1f}% |")

        lines.append("\n### Potential Samples by Error Count (Top 15)\n")
        lines.append("| Error Count | Total Potential Samples |")
        lines.append("|-------------|-------------------------|")
        for error_count, total_samples in exhaust['potential_by_error_count'].items():
            lines.append(f"| {error_count} | {int(total_samples):,} |")

        lines.append("\n### Snippet Length Distribution (Top 30)\n")
        lines.append("| Length (words) | Examples |")
        lines.append("|----------------|----------|")
        for length, count in exhaust['snippet_length_dist'].items():
            lines.append(f"| {length} | {count:,} |")

        # Error Patterns
        patterns = self.stats['error_patterns']
        lines.append("\n---\n")
        lines.append("## 8. Error Pattern Analysis\n")
        lines.append("| Error Category | Count | Percentage |")
        lines.append("|----------------|-------|------------|")
        total_instances = errors['total_instances']
        # Sort by count descending
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_patterns:
            pct = 100 * count / total_instances if total_instances else 0
            lines.append(f"| {category.replace('_', ' ').title()} | {count:,} | {pct:.1f}% |")

        # Key Insights
        lines.append("\n---\n")
        lines.append("## 9. Key Insights & Recommendations\n")
        lines.append("\n### Dataset Characteristics\n")
        lines.append(f"- ✅ **Size**: {basic['total_examples']:,} examples")
        lines.append(f"- 📊 **Error diversity**: {errors['total_instances']:,} error instances across {errors['total_errors']:,} targets")
        lines.append(f"- 🎯 **Avg instances per error**: {errors['avg_instances_per_error']:.1f}")

        lines.append("\n### Recommendations for Augmentation\n")

        # Find most common error count
        most_common_error_count = max(dist['error_count_dist'].items(), key=lambda x: x[1])[0]
        lines.append(f"1. **Most examples have {most_common_error_count} errors** - exhaustive augmentation feasible for these")

        # Find most common instance count
        most_common_instances = max(dist['instance_count_dist'].items(), key=lambda x: x[1])[0]
        lines.append(f"2. **Most errors have {most_common_instances} instance(s)** - consider in combination calculations")

        # Avg error density
        avg_density = errors['avg_errors_per_example'] / basic['avg_snippet_length'] * 100
        lines.append(f"3. **Average error density: {avg_density:.1f}%** - aligns with 5-30% augmentation range")

        # Top error types
        top_error_type = sorted_patterns[0][0].replace('_', ' ')
        lines.append(f"4. **Dominant error type: {top_error_type}** - ensure adequate representation")

        lines.append(f"\n5. **Exhaustive strategy recommendation**: Most examples can be fully exhaustively generated")

        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"✅ Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced EDA for Revita data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--data-file',
        type=Path,
        default=Path('data/revita/exercise_errors_Finnish.jsonl'),
        help='Input Revita JSONL file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('reports/revita_eda.md'),
        help='Output markdown report'
    )

    args = parser.parse_args()

    # Run analysis
    eda = RevitaEDA(args.data_file)
    eda.load_data()
    eda.analyze_all()
    eda.generate_report(args.output)

    print(f"\n✨ EDA complete!")


if __name__ == '__main__':
    main()
