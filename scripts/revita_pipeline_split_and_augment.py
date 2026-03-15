#!/usr/bin/env python3
"""
Pipeline: Split raw data + Augment train/val/test separately.

This script orchestrates the complete workflow:
1. Split raw data into train/val/test (BEFORE augmentation)
2. Augment training data
3. Augment validation data
4. Augment test data
5. Generate EDA reports for all splits

This ensures no data leakage between train, val, and test sets.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*70}")
    print(f"🔧 {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n✅ {description} completed successfully")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Complete pipeline: Split raw data → Augment train/val/test separately',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default pipeline with stratified split and random augmentation (70/10/20)
  python revita_pipeline_split_and_augment.py

  # Custom ratios (80/10/10)
  python revita_pipeline_split_and_augment.py --val-ratio 0.1 --test-ratio 0.1

  # Exhaustive augmentation with limit
  python revita_pipeline_split_and_augment.py --strategy exhaustive --max-augmentation 500

  # Simple random split (no stratification)
  python revita_pipeline_split_and_augment.py --stratify none

  # Skip EDA report generation
  python revita_pipeline_split_and_augment.py --skip-eda
        """
    )

    # Input/output paths
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/revita/exercise_errors_Finnish_cleaned.jsonl'),
        help='Input cleaned raw JSONL file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/revita'),
        help='Output directory for all generated files'
    )

    # Split parameters
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
        default='error_count',
        help='Stratification strategy (default: error_count)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for split reproducibility (default: 42)'
    )

    # Augmentation parameters
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['random', 'exhaustive'],
        default='random',
        help='Augmentation strategy (default: random)'
    )
    parser.add_argument(
        '--max-error-rate',
        type=float,
        default=0.2,
        help='Maximum error density (default: 0.2 = 20%%)'
    )
    parser.add_argument(
        '--max-augmentation',
        type=int,
        default=None,
        help='Max augmented samples per raw example (default: None = greedy)'
    )
    parser.add_argument(
        '--no-clean',
        action='store_true',
        help='Exclude clean (correct→correct) samples'
    )
    parser.add_argument(
        '--augment-seed',
        type=int,
        default=42,
        help='Random seed for augmentation (default: 42, test uses seed+1)'
    )

    # Pipeline options
    parser.add_argument(
        '--skip-eda',
        action='store_true',
        help='Skip EDA report generation'
    )
    parser.add_argument(
        '--skip-split',
        action='store_true',
        help='Skip split step (use existing train_raw.jsonl, val_raw.jsonl, and test_raw.jsonl)'
    )

    args = parser.parse_args()

    # Validate
    if args.val_ratio <= 0 or args.val_ratio >= 1:
        parser.error("val-ratio must be between 0 and 1")
    if args.test_ratio <= 0 or args.test_ratio >= 1:
        parser.error("test-ratio must be between 0 and 1")
    if args.val_ratio + args.test_ratio >= 1:
        parser.error("val-ratio + test-ratio must be less than 1")
    if args.max_error_rate <= 0 or args.max_error_rate > 1:
        parser.error("max-error-rate must be between 0 and 1")

    # Paths
    splits_dir = args.output_dir / 'splits'
    train_raw_path = splits_dir / 'train_raw.jsonl'
    val_raw_path = splits_dir / 'val_raw.jsonl'
    test_raw_path = splits_dir / 'test_raw.jsonl'

    # Build descriptive suffix for output filenames
    suffix_parts = [args.strategy]
    if args.max_augmentation:
        suffix_parts.append(f'limit{args.max_augmentation}')
    else:
        suffix_parts.append('greedy')
    suffix_parts.append(f'errdensity{int(args.max_error_rate * 100)}')
    suffix_parts.append('clean' if not args.no_clean else 'noclean')
    suffix_parts.append(f'seed{args.augment_seed}')
    suffix = '_'.join(suffix_parts)

    # Output augmented files to splits directory for better organization
    train_augmented_path = splits_dir / f'train_augmented_{suffix}.jsonl'
    val_augmented_path = splits_dir / f'val_augmented_{suffix}.jsonl'
    test_augmented_path = splits_dir / f'test_augmented_{suffix}.jsonl'

    # Start pipeline
    train_ratio = 1 - args.val_ratio - args.test_ratio
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*70}")
    print(f"🚀 REVITA DATA PIPELINE")
    print(f"{'='*70}")
    print(f"Started: {timestamp}")
    print(f"\nInput: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nPipeline configuration:")
    print(f"  Split strategy: {args.stratify if args.stratify != 'none' else 'simple random'}")
    print(f"  Train/Val/Test ratio: {train_ratio*100:.0f}% / {args.val_ratio*100:.0f}% / {args.test_ratio*100:.0f}%")
    print(f"  Augmentation: {args.strategy}")
    print(f"  Max error rate: {args.max_error_rate * 100:.0f}%")
    if args.max_augmentation:
        print(f"  Max samples/example: {args.max_augmentation}")
    print(f"  Include clean samples: {not args.no_clean}")
    print(f"{'='*70}")

    # Step 1: Split raw data
    if not args.skip_split:
        split_cmd = [
            'python', 'scripts/revita_split_clean_raw_data.py',
            '--input', str(args.input),
            '--output-dir', str(splits_dir),
            '--val-ratio', str(args.val_ratio),
            '--test-ratio', str(args.test_ratio),
            '--stratify', args.stratify,
            '--seed', str(args.seed)
        ]
        run_command(split_cmd, "Step 1: Split raw data into train/val/test")
    else:
        print(f"\n⏭️  Skipping split step (using existing files)")
        if not train_raw_path.exists() or not val_raw_path.exists() or not test_raw_path.exists():
            print(f"❌ Error: --skip-split specified but required files not found:")
            print(f"   {train_raw_path}")
            print(f"   {val_raw_path}")
            print(f"   {test_raw_path}")
            sys.exit(1)

    # Step 2: Augment training data
    train_augment_cmd = [
        'python', 'scripts/revita_augment_raw_data.py',
        '--input', str(train_raw_path),
        '--output', str(train_augmented_path),
        '--strategy', args.strategy,
        '--max-error-rate', str(args.max_error_rate),
        '--seed', str(args.augment_seed)
    ]
    if args.max_augmentation:
        train_augment_cmd.extend(['--max-augmentation-per-raw-example', str(args.max_augmentation)])
    if args.no_clean:
        train_augment_cmd.append('--no-clean')

    run_command(train_augment_cmd, "Step 2: Augment training data")

    # Step 3: Augment validation data (different seed to avoid identical augmentations)
    val_augment_cmd = [
        'python', 'scripts/revita_augment_raw_data.py',
        '--input', str(val_raw_path),
        '--output', str(val_augmented_path),
        '--strategy', args.strategy,
        '--max-error-rate', str(args.max_error_rate),
        '--seed', str(args.augment_seed + 1)  # Different seed for val
    ]
    if args.max_augmentation:
        val_augment_cmd.extend(['--max-augmentation-per-raw-example', str(args.max_augmentation)])
    if args.no_clean:
        val_augment_cmd.append('--no-clean')

    run_command(val_augment_cmd, "Step 3: Augment validation data")

    # Step 4: Augment test data (different seed to avoid identical augmentations)
    test_augment_cmd = [
        'python', 'scripts/revita_augment_raw_data.py',
        '--input', str(test_raw_path),
        '--output', str(test_augmented_path),
        '--strategy', args.strategy,
        '--max-error-rate', str(args.max_error_rate),
        '--seed', str(args.augment_seed + 2)  # Different seed for test
    ]
    if args.max_augmentation:
        test_augment_cmd.extend(['--max-augmentation-per-raw-example', str(args.max_augmentation)])
    if args.no_clean:
        test_augment_cmd.append('--no-clean')

    run_command(test_augment_cmd, "Step 4: Augment test data")

    # Step 5: Generate EDA reports (optional)
    if not args.skip_eda:
        reports_dir = Path('reports')

        # Train EDA
        train_eda_cmd = [
            'python', 'scripts/revita_eda_augmented_data.py',
            '--input', str(train_augmented_path),
            '--output', str(reports_dir / f'train_augmented_{suffix}_eda.md')
        ]
        run_command(train_eda_cmd, "Step 5a: Generate training data EDA report")

        # Val EDA
        val_eda_cmd = [
            'python', 'scripts/revita_eda_augmented_data.py',
            '--input', str(val_augmented_path),
            '--output', str(reports_dir / f'val_augmented_{suffix}_eda.md')
        ]
        run_command(val_eda_cmd, "Step 5b: Generate validation data EDA report")

        # Test EDA
        test_eda_cmd = [
            'python', 'scripts/revita_eda_augmented_data.py',
            '--input', str(test_augmented_path),
            '--output', str(reports_dir / f'test_augmented_{suffix}_eda.md')
        ]
        run_command(test_eda_cmd, "Step 5c: Generate test data EDA report")

    # Pipeline complete
    print(f"\n{'='*70}")
    print(f"✅ PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"\n📂 Raw splits:")
    print(f"   Train: {train_raw_path}")
    print(f"   Val:   {val_raw_path}")
    print(f"   Test:  {test_raw_path}")
    print(f"\n📊 Augmented data:")
    print(f"   Train: {train_augmented_path}")
    print(f"   Val:   {val_augmented_path}")
    print(f"   Test:  {test_augmented_path}")
    if not args.skip_eda:
        print(f"\n📈 EDA reports:")
        print(f"   Train: reports/train_augmented_{suffix}_eda.md")
        print(f"   Val:   reports/val_augmented_{suffix}_eda.md")
        print(f"   Test:  reports/test_augmented_{suffix}_eda.md")

    print(f"\n{'='*70}")
    print(f"Next steps:")
    print(f"  1. Review EDA reports to verify data quality")
    print(f"  2. Update model configs with correct data paths")
    print(f"  3. Train your GEC model on: {train_augmented_path}")
    print(f"  4. Validate on: {val_augmented_path}")
    print(f"  5. Evaluate on: {test_augmented_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
