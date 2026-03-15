#!/usr/bin/env python3
"""
Augment Revita data into concrete training samples for GEC.

Transforms raw Revita format (snippet + errors with instances) into
standard (corrupted, correct) pairs for seq2seq training.

Strategies:
- Random sampling: Generate N samples per error count (default)
- Exhaustive: Generate ALL combinations of errors and instances

Features:
- Generate ALL error counts from 1 to max_errors
- Control error density (tunable max error rate)
- Utilize all error instances across augmented samples
- Include clean samples (correct→correct) to prevent overcorrection by default
"""

import json
import argparse
import random
import itertools
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class RevitaAugmentor:
    """Augment Revita data into training samples."""

    def __init__(
        self,
        max_error_rate: float = 0.2,
        max_augmentation_per_raw_example: Optional[int] = None,
        include_clean: bool = True,
        strategy: str = 'random',
        seed: int = 42
    ):
        """
        Initialize augmentor.

        Args:
            max_error_rate: Maximum error density (0.2 = max 20% of snippet)
            max_augmentation_per_raw_example: Max samples per original (None = greedy/exhaustive)
            include_clean: Whether to include clean (correct→correct) samples (default: True)
            strategy: 'random' for random sampling or 'exhaustive' for all combinations (default: random)
            seed: Random seed for reproducibility
        """
        self.max_error_rate = max_error_rate
        self.max_augmentation_per_raw_example = max_augmentation_per_raw_example
        self.include_clean = include_clean
        self.strategy = strategy
        self.seed = seed
        random.seed(seed)

        self.stats = defaultdict(int)

    def clean_text(self, text: str) -> str:
        """
        Clean text for NLP processing.

        - Strip leading/trailing whitespace
        - Replace newlines with spaces
        - Replace multiple spaces with single space
        """
        # Strip leading/trailing whitespace
        text = text.strip()

        # Replace newlines with spaces
        text = text.replace('\n', ' ')

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        return text

    def snippet_to_text(self, snippet: List[str]) -> str:
        """Convert snippet tokens to clean text."""
        # Join tokens
        text = ''.join(snippet)
        # Clean the text
        text = self.clean_text(text)
        return text

    def apply_errors(
        self,
        snippet: List[str],
        errors_to_apply: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        Apply selected errors to snippet.

        Args:
            snippet: Original correct tokens
            errors_to_apply: List of error dicts with 'wid', 'word', 'instance'

        Returns:
            (corrupted_text, applied_edits_metadata)
        """
        # Create a copy of snippet
        corrupted = snippet.copy()
        applied_edits = []

        # Sort errors by position (descending) to avoid index shifting
        errors_sorted = sorted(errors_to_apply, key=lambda e: e['wid'][0], reverse=True)

        for error in errors_sorted:
            wid = error['wid']
            correct_word = error['word']
            incorrect_instance = error['instance']

            # Handle multi-word errors (wid can be a list of positions)
            if len(wid) == 1:
                # Single word error
                pos = wid[0]
                if pos < len(corrupted):
                    original = corrupted[pos]
                    corrupted[pos] = incorrect_instance

                    applied_edits.append({
                        'position': pos,
                        'correct': correct_word,
                        'corrupted': incorrect_instance,
                        'original_token': original
                    })
            else:
                # Multi-word error (e.g., compound)
                start_pos = min(wid)

                # Store original
                original_span = ' '.join([corrupted[i] for i in wid if i < len(corrupted)])

                # Replace first position with error, remove others
                if start_pos < len(corrupted):
                    corrupted[start_pos] = incorrect_instance
                    # Mark other positions for removal
                    for i in sorted(wid[1:], reverse=True):
                        if i < len(corrupted):
                            corrupted.pop(i)

                    applied_edits.append({
                        'position': start_pos,
                        'correct': correct_word,
                        'corrupted': incorrect_instance,
                        'original_span': original_span,
                        'multi_word': True
                    })

        # Convert to text and clean
        corrupted_text = self.snippet_to_text(corrupted)
        return corrupted_text, applied_edits

    def _generate_random_samples(
        self,
        snippet: List[str],
        errors: List[Dict],
        error_counts: List[int],
        correct_text: str,
        snippet_length: int,
        augmented_samples: List[Dict]
    ):
        """Generate samples using random sampling strategy."""
        if self.max_augmentation_per_raw_example:
            # Limited mode: distribute budget across error counts
            samples_per_count = max(1, self.max_augmentation_per_raw_example // len(error_counts))
        else:
            # Greedy mode: generate more samples per error count
            samples_per_count = 10

        for num_errors in error_counts:
            # Generate multiple samples for this specific error count
            for _ in range(samples_per_count):
                # Randomly select which errors to apply
                if num_errors <= len(errors):
                    selected_errors = random.sample(errors, num_errors)

                    # For each selected error, randomly pick an instance
                    errors_to_apply = []
                    for error in selected_errors:
                        instances = error.get('instances', [])
                        if instances:
                            selected_instance = random.choice(instances)
                            errors_to_apply.append({
                                'wid': error['wid'],
                                'word': error['word'],
                                'instance': selected_instance
                            })

                    # Apply the errors
                    if errors_to_apply:
                        corrupted_text, edits = self.apply_errors(snippet, errors_to_apply)

                        # Only add if corruption actually changed the text
                        if corrupted_text != correct_text:
                            error_rate = len(edits) / snippet_length if snippet_length > 0 else 0
                            augmented_samples.append({
                                'corrupted': corrupted_text,
                                'correct': correct_text,
                                'num_errors': len(edits),
                                'snippet_length': snippet_length,
                                'error_rate': error_rate,
                                'edits': edits,
                                'source': 'revita_augmented'
                            })
                            self.stats['augmented_samples'] += 1
                            self.stats[f'errors_{len(edits)}'] += 1

                # Check if we've hit the limit
                if self.max_augmentation_per_raw_example:
                    if len(augmented_samples) >= self.max_augmentation_per_raw_example + (1 if self.include_clean else 0):
                        break

            # Check limit again at outer loop
            if self.max_augmentation_per_raw_example:
                if len(augmented_samples) >= self.max_augmentation_per_raw_example + (1 if self.include_clean else 0):
                    break

    def _generate_exhaustive_samples(
        self,
        snippet: List[str],
        errors: List[Dict],
        error_counts: List[int],
        correct_text: str,
        snippet_length: int,
        augmented_samples: List[Dict]
    ):
        """Generate samples using exhaustive combination strategy."""
        for num_errors in error_counts:
            if num_errors > len(errors):
                continue

            # Generate all combinations of error selections
            for error_combo in itertools.combinations(errors, num_errors):
                # For each combination, generate all instance combinations
                instance_lists = []
                for error in error_combo:
                    instances = error.get('instances', [])
                    if instances:
                        instance_lists.append([
                            {
                                'wid': error['wid'],
                                'word': error['word'],
                                'instance': inst
                            }
                            for inst in instances
                        ])

                # Generate all combinations of instances
                if instance_lists:
                    for instance_combo in itertools.product(*instance_lists):
                        errors_to_apply = list(instance_combo)

                        # Apply the errors
                        corrupted_text, edits = self.apply_errors(snippet, errors_to_apply)

                        # Only add if corruption actually changed the text
                        if corrupted_text != correct_text:
                            error_rate = len(edits) / snippet_length if snippet_length > 0 else 0
                            augmented_samples.append({
                                'corrupted': corrupted_text,
                                'correct': correct_text,
                                'num_errors': len(edits),
                                'snippet_length': snippet_length,
                                'error_rate': error_rate,
                                'edits': edits,
                                'source': 'revita_augmented_exhaustive'
                            })
                            self.stats['augmented_samples'] += 1
                            self.stats[f'errors_{len(edits)}'] += 1

                        # Check if we've hit the limit
                        if self.max_augmentation_per_raw_example:
                            if len(augmented_samples) >= self.max_augmentation_per_raw_example + (1 if self.include_clean else 0):
                                return

    def generate_augmented_samples(
        self,
        example: Dict
    ) -> List[Dict]:
        """
        Generate augmented samples from one original example.

        For each error count from min to max:
        - Generate multiple samples with that error count
        - Randomly select which errors and which instances

        Args:
            example: Original Revita example with snippet and errors

        Returns:
            List of augmented samples (corrupted, correct, metadata)
        """
        snippet = example['snippet']
        errors = example['errors']
        correct_text = self.snippet_to_text(snippet)

        # Count actual words (not whitespace)
        words = [t for t in snippet if t.strip() and t not in [' ', '\n\n']]
        snippet_length = len(words)

        # Calculate error range based on snippet length
        min_errors = 1  # Always start from 1 error
        max_errors = max(1, int(snippet_length * self.max_error_rate))
        max_errors = min(max_errors, len(errors))  # Can't exceed available errors

        augmented_samples = []

        # Generate clean sample (0 errors) if enabled
        if self.include_clean:
            augmented_samples.append({
                'corrupted': correct_text,
                'correct': correct_text,
                'num_errors': 0,
                'snippet_length': snippet_length,
                'error_rate': 0.0,
                'edits': [],
                'source': 'revita_clean'
            })
            self.stats['clean_samples'] += 1

        # Generate samples based on strategy
        error_counts = list(range(min_errors, max_errors + 1))

        if self.strategy == 'exhaustive':
            self._generate_exhaustive_samples(
                snippet, errors, error_counts, correct_text,
                snippet_length, augmented_samples
            )
        else:  # random
            self._generate_random_samples(
                snippet, errors, error_counts, correct_text,
                snippet_length, augmented_samples
            )

        return augmented_samples

    def augment_dataset(
        self,
        input_path: Path,
        output_path: Path,
        max_examples: int = None
    ):
        """
        Augment entire Revita dataset.

        Args:
            input_path: Path to raw Revita JSONL file
            output_path: Path to output augmented JSONL file
            max_examples: Optional limit on input examples to process
        """
        print(f"📂 Loading data from {input_path}")

        examples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                if line.strip():
                    examples.append(json.loads(line))

        print(f"✓ Loaded {len(examples)} examples")
        print(f"\n🔄 Generating augmented samples...")
        print(f"   Strategy: {self.strategy.upper()}")
        print(f"   Max error rate: {self.max_error_rate * 100:.0f}%")
        if self.max_augmentation_per_raw_example:
            print(f"   Max augmentation per example: {self.max_augmentation_per_raw_example}")
        else:
            mode_desc = "all combinations" if self.strategy == 'exhaustive' else "10 samples per error count"
            print(f"   Mode: Greedy ({mode_desc})")
        print(f"   Include clean samples: {self.include_clean}")

        # Generate augmented samples
        all_augmented = []
        for i, example in enumerate(examples):
            augmented = self.generate_augmented_samples(example)
            all_augmented.extend(augmented)

            if (i + 1) % 500 == 0:
                print(f"   Processed {i + 1}/{len(examples)} examples...")

        # Shuffle to mix error densities
        random.shuffle(all_augmented)

        # Add training weights based on correct text frequency
        print(f"\n⚖️  Calculating training weights...")
        correct_text_counts = defaultdict(int)

        # Count frequency of each unique correct text
        for sample in all_augmented:
            correct_text_counts[sample['correct']] += 1

        # Add inverse frequency weight to each sample
        for sample in all_augmented:
            count = correct_text_counts[sample['correct']]
            sample['training_weight'] = 1.0 / count
            sample['correct_frequency'] = count

        # Statistics on text frequency distribution
        frequencies = list(correct_text_counts.values())
        unique_texts = len(correct_text_counts)
        max_freq = max(frequencies) if frequencies else 0
        min_freq = min(frequencies) if frequencies else 0
        avg_freq = sum(frequencies) / len(frequencies) if frequencies else 0

        print(f"   Unique correct texts: {unique_texts:,}")
        print(f"   Frequency range: Min={min_freq}, Max={max_freq}, Avg={avg_freq:.1f}")
        print(f"   Most repeated text: {max_freq}× (weight={1.0/max_freq:.4f})")
        print(f"   Least repeated text: {min_freq}× (weight={1.0/min_freq:.4f})")

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in all_augmented:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # Calculate statistics
        if all_augmented:
            avg_error_rate = sum(s.get('error_rate', 0) for s in all_augmented) / len(all_augmented)
            avg_snippet_length = sum(s.get('snippet_length', 0) for s in all_augmented) / len(all_augmented)
        else:
            avg_error_rate = 0
            avg_snippet_length = 0

        # Print statistics
        print(f"\n✅ Augmentation complete!")
        print(f"   Input examples: {len(examples):,}")
        print(f"   Output samples: {len(all_augmented):,}")
        print(f"   Expansion factor: {len(all_augmented) / len(examples):.1f}x")
        print(f"   Avg snippet length: {avg_snippet_length:.1f} words")
        print(f"   Avg error rate: {avg_error_rate * 100:.1f}%")
        print(f"\n📊 Sample breakdown:")
        print(f"   Clean samples: {self.stats['clean_samples']:,}")
        print(f"   Augmented samples: {self.stats['augmented_samples']:,}")
        print(f"\n   By error count:")
        error_keys = sorted([k for k in self.stats.keys() if k.startswith('errors_')],
                          key=lambda x: int(x.split('_')[1]))
        for key in error_keys[:20]:  # Show top 20
            count = self.stats[key]
            num = int(key.split('_')[1])
            pct = 100 * count / len(all_augmented) if all_augmented else 0
            print(f"     {num} error(s): {count:,} ({pct:.1f}%)")

        print(f"\n💾 Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Augment Revita data for GEC training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: random sampling with max 20% error rate
  python augment_revita_data.py

  # Exhaustive generation (all combinations)
  python augment_revita_data.py --strategy exhaustive

  # Exhaustive with max limit (prevents huge datasets)
  python augment_revita_data.py --strategy exhaustive --max-augmentation-per-raw-example 500

  # Custom error density
  python augment_revita_data.py --max-error-rate 0.3

  # Without clean samples
  python augment_revita_data.py --no-clean
        """
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/revita/exercise_errors_Finnish_cleaned.jsonl'),
        help='Input cleaned Revita JSONL file (default: data/revita/exercise_errors_Finnish_cleaned.jsonl)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/revita/augmented_Finnish.jsonl'),
        help='Output augmented JSONL file (default: data/revita/augmented_Finnish.jsonl)'
    )
    parser.add_argument(
        '--max-error-rate',
        type=float,
        default=0.2,
        help='Maximum error density: 0.2 = max 20%% of snippet (default: 0.2)'
    )
    parser.add_argument(
        '--max-augmentation-per-raw-example',
        type=int,
        default=None,
        help='Max augmented samples per original (default: None = greedy mode)'
    )
    parser.add_argument(
        '--no-clean',
        action='store_true',
        help='Exclude clean (correct→correct) samples (default: include them)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['random', 'exhaustive'],
        default='random',
        help='Sampling strategy: "random" for random sampling or "exhaustive" for all combinations (default: random)'
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        help='Limit number of input examples to process (for testing)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Validate error rate
    if args.max_error_rate <= 0 or args.max_error_rate > 1:
        parser.error("max-error-rate must be between 0 and 1")

    # Auto-generate output filename if using default
    if args.output == Path('data/revita/augmented_Finnish.jsonl') or str(args.output) == 'data/revita/augmented_Finnish.jsonl':
        # Build filename from parameters
        parts = ['revita_augmented']

        # Strategy
        parts.append(args.strategy)

        # Max augmentation limit
        if args.max_augmentation_per_raw_example:
            parts.append(f'limit{args.max_augmentation_per_raw_example}')
        else:
            parts.append('greedy')

        # Error rate
        max_err = int(args.max_error_rate * 100)
        parts.append(f'errdensity{max_err}')

        # Clean samples
        if not args.no_clean:
            parts.append('clean')
        else:
            parts.append('noclean')

        # Seed
        parts.append(f'seed{args.seed}')

        # Construct filename
        filename = '_'.join(parts) + '.jsonl'
        args.output = Path('data/revita') / filename

        print(f"📝 Auto-generated output filename: {args.output}")

    # Create augmentor
    augmentor = RevitaAugmentor(
        max_error_rate=args.max_error_rate,
        max_augmentation_per_raw_example=args.max_augmentation_per_raw_example,
        include_clean=not args.no_clean,  # Default is True
        strategy=args.strategy,
        seed=args.seed
    )

    # Augment dataset
    augmentor.augment_dataset(
        input_path=args.input,
        output_path=args.output,
        max_examples=args.max_examples
    )


if __name__ == '__main__':
    main()
