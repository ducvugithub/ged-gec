#!/usr/bin/env python3
"""
Clean Revita raw data by filtering out problematic error instances.

Filters out:
1. Instances with emojis
2. Only-number instances (e.g., "10") but keeps mixed (e.g., "13:aan", "15-vuotias")
3. Single character instances (e.g., "E")
4. Blacklisted instances (Russian words, obvious English words)

Usage:
    python scripts/revita_clean_raw_samples.py \
        --input data/revita/exercise_errors_Finnish.jsonl \
        --output data/revita/exercise_errors_Finnish_cleaned.jsonl
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict

def has_emoji(text: str) -> bool:
    """Check if text contains emoji characters."""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return bool(emoji_pattern.search(text))


def is_only_number(text: str) -> bool:
    """
    Check if instance is ONLY a number.

    Returns True for: "10", "123"
    Returns False for: "13:aan", "15-vuotias", "1800-luku"
    """
    return bool(re.match(r'^\d+$', text.strip()))


def is_single_char(text: str) -> bool:
    """Check if instance is a single character."""
    return len(text.strip()) == 1


# Hardcoded blacklist of specific problematic instances
BLACKLIST_INSTANCES = {
    # Russian words
    'Они',
    'быть',
    'лодка',
    'лодки',
    'лодок',
    'недели',
    'неделя',
    'этл',
    'это',

    # English words (only obvious ones, not Finnish false positives)
    'She',
    'need to',
    'forestry work',
    'heard',
    'known',
    'interests',
    'this',
    'that',
    'where',
    'choose',
    'chosen',
    'is located',
    'is performed',
    'is doing',
    'fourth',
    'from the year',
    'what',
    'think',
    'Competition',
    'countries',
    'country',
    'listening practice',
    'hat',
    'headgear',
    'child',
    'whi',
}


def is_blacklisted(text: str) -> bool:
    """Check if instance is in the hardcoded blacklist."""
    return text in BLACKLIST_INSTANCES


def should_filter_instance(instance: str) -> tuple[bool, str]:
    """
    Check if instance should be filtered out.

    Returns:
        (should_filter: bool, reason: str)
    """
    if is_blacklisted(instance):
        return (True, 'blacklisted')

    if has_emoji(instance):
        return (True, 'emoji')

    if is_only_number(instance):
        return (True, 'only_number')

    if is_single_char(instance):
        return (True, 'single_char')

    # English detection disabled - too many false positives
    # Only use hardcoded blacklist for English instances
    # if is_english(instance):
    #     return (True, 'english')

    return (False, '')


def clean_raw_data(
    input_path: Path,
    output_path: Path,
    log_filtered: bool = True
):
    """
    Clean raw Revita data by filtering problematic instances.

    Args:
        input_path: Path to raw Revita JSONL
        output_path: Path to save cleaned JSONL
        log_filtered: Whether to log filtered instances
    """
    print(f"📂 Loading data from {input_path}")

    # Load all examples
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"✓ Loaded {len(examples)} examples")

    # Statistics
    stats = defaultdict(int)
    filtered_log = []

    # Clean each example
    cleaned_examples = []

    for example in examples:
        snippet = example['snippet']
        errors = example.get('errors', [])

        cleaned_errors = []

        for error in errors:
            wid = error['wid']
            word = error['word']
            instances = error.get('instances', [])

            # Filter instances
            cleaned_instances = []

            for inst in instances:
                should_filter, reason = should_filter_instance(inst)

                if should_filter:
                    stats[f'filtered_{reason}'] += 1
                    stats['total_filtered'] += 1

                    if log_filtered:
                        filtered_log.append({
                            'instance': inst,
                            'reason': reason,
                            'correct_word': word
                        })
                else:
                    cleaned_instances.append(inst)
                    stats['kept'] += 1

            # Only keep error if it still has instances after filtering
            if cleaned_instances:
                cleaned_errors.append({
                    'wid': wid,
                    'word': word,
                    'instances': cleaned_instances
                })
            else:
                stats['errors_fully_filtered'] += 1

        # Only keep example if it still has errors after filtering
        if cleaned_errors:
            cleaned_examples.append({
                'snippet': snippet,
                'errors': cleaned_errors
            })
        else:
            stats['examples_fully_filtered'] += 1

    # Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for example in cleaned_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    # Save filtered instances log
    if log_filtered and filtered_log:
        log_path = output_path.parent / f"{output_path.stem}_filtered_log.jsonl"
        with open(log_path, 'w', encoding='utf-8') as f:
            for item in filtered_log:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"📝 Filtered instances logged to: {log_path}")

    # Print statistics
    print(f"\n✅ Cleaning complete!")
    print(f"   Input examples: {len(examples):,}")
    print(f"   Output examples: {len(cleaned_examples):,}")
    print(f"   Examples fully filtered: {stats['examples_fully_filtered']:,}")

    print(f"\n📊 Instance statistics:")
    print(f"   Total instances kept: {stats['kept']:,}")
    print(f"   Total instances filtered: {stats['total_filtered']:,}")

    if stats['total_filtered'] > 0:
        print(f"\n   Filtered breakdown:")
        if stats.get('filtered_blacklisted', 0) > 0:
            print(f"     - Blacklisted: {stats['filtered_blacklisted']:,}")
        if stats.get('filtered_emoji', 0) > 0:
            print(f"     - With emoji: {stats['filtered_emoji']:,}")
        if stats.get('filtered_only_number', 0) > 0:
            print(f"     - Only numbers: {stats['filtered_only_number']:,}")
        if stats.get('filtered_single_char', 0) > 0:
            print(f"     - Single character: {stats['filtered_single_char']:,}")
        if stats.get('filtered_english', 0) > 0:
            print(f"     - English: {stats['filtered_english']:,}")

    if stats.get('errors_fully_filtered', 0) > 0:
        print(f"\n   Errors fully filtered: {stats['errors_fully_filtered']:,}")

    print(f"\n💾 Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Clean Revita raw data by filtering problematic instances',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Filters out:
  1. Instances with emojis
  2. Only-number instances (e.g., "10") but keeps mixed (e.g., "13:aan")
  3. Single character instances (e.g., "E")
  4. English instances (detected via langdetect)

Examples:
  # Clean raw data
  python scripts/revita_clean_raw_samples.py \\
      --input data/revita/exercise_errors_Finnish.jsonl \\
      --output data/revita/exercise_errors_Finnish_cleaned.jsonl

  # Then augment the cleaned data
  python scripts/augment_revita_data.py \\
      --input data/revita/exercise_errors_Finnish_cleaned.jsonl \\
      --strategy exhaustive --max-augmentation-per-raw-example 500
        """
    )

    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/revita/exercise_errors_Finnish.jsonl'),
        help='Input raw Revita JSONL file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/revita/exercise_errors_Finnish_cleaned.jsonl'),
        help='Output cleaned JSONL file'
    )
    parser.add_argument(
        '--no-log',
        action='store_true',
        help='Do not log filtered instances'
    )

    args = parser.parse_args()

    clean_raw_data(
        input_path=args.input,
        output_path=args.output,
        log_filtered=not args.no_log
    )


if __name__ == '__main__':
    main()
