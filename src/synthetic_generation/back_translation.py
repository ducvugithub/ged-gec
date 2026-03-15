"""
Back-translation corruption: Finnish → X → Finnish via MT.

Surface differences become synthetic errors.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, List


class BackTranslationCorruptor:
    """Generate corrupted sentences via back-translation."""

    def __init__(self, pivot_lang: str = 'en'):
        """
        Initialize back-translation corruptor.

        Args:
            pivot_lang: Pivot language for back-translation (e.g., 'en', 'sv')
        """
        self.pivot_lang = pivot_lang
        # TODO: Initialize translation models
        # e.g., Helsinki-NLP/opus-mt-fi-en and opus-mt-en-fi

    def back_translate(self, sentence: str) -> str:
        """
        Back-translate a sentence: Finnish → pivot → Finnish.

        Args:
            sentence: Original Finnish sentence

        Returns:
            Back-translated Finnish sentence
        """
        # TODO: Implement translation pipeline
        # 1. Translate Finnish → pivot language
        # 2. Translate pivot → Finnish
        return sentence

    def process_file(self, input_path: Path, output_path: Path):
        """Process input file and generate back-translated corruptions."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:

            for line in f_in:
                sentence = line.strip()
                if not sentence:
                    continue

                corrupted = self.back_translate(sentence)

                # Only save if back-translation differs from original
                if corrupted != sentence:
                    data = {
                        'corrupted': corrupted,
                        'correct': sentence,
                        'source': f'back_translation_{self.pivot_lang}'
                    }
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Generate back-translation corruptions')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pivot-lang', type=str, default='en', help='Pivot language')

    args = parser.parse_args()

    corruptor = BackTranslationCorruptor(pivot_lang=args.pivot_lang)
    corruptor.process_file(args.input, args.output)

    print(f"✓ Generated back-translation corruptions: {args.output}")


if __name__ == '__main__':
    main()
