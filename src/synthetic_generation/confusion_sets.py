"""
Confusion set-based corruption for Finnish GEC.

Uses UD analyser to generate linguistically-aware morphological confusions
(e.g., partitive vs. accusative, past vs. present).
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import random


class ConfusionSetCorruptor:
    """Generate corrupted Finnish sentences using confusion sets."""

    def __init__(self, corruption_rate: float = 0.15):
        """
        Initialize the confusion set corruptor.

        Args:
            corruption_rate: Probability of corrupting each token (0.0-1.0)
        """
        self.corruption_rate = corruption_rate
        # TODO: Initialize UD analyser here

    def build_confusion_set(self, token: str) -> List[str]:
        """
        Build a confusion set for a given token using morphological analysis.

        Args:
            token: Input token to build confusion set for

        Returns:
            List of plausible confusable forms
        """
        # TODO: Use UD analyser to generate morphological variants
        # Example confusion types:
        # - Case confusion: nominative ↔ partitive ↔ accusative
        # - Tense confusion: present ↔ past ↔ perfect
        # - Agreement: singular ↔ plural
        return []

    def corrupt_sentence(self, sentence: str) -> Tuple[str, str, List[Dict]]:
        """
        Corrupt a clean sentence using confusion sets.

        Args:
            sentence: Clean input sentence

        Returns:
            Tuple of (corrupted_sentence, original_sentence, edits_metadata)
        """
        tokens = sentence.split()
        corrupted_tokens = []
        edits = []

        for i, token in enumerate(tokens):
            if random.random() < self.corruption_rate:
                confusion_set = self.build_confusion_set(token)
                if confusion_set:
                    corrupted = random.choice(confusion_set)
                    corrupted_tokens.append(corrupted)
                    edits.append({
                        'position': i,
                        'original': token,
                        'corrupted': corrupted,
                        'type': 'confusion_set'
                    })
                else:
                    corrupted_tokens.append(token)
            else:
                corrupted_tokens.append(token)

        return ' '.join(corrupted_tokens), sentence, edits

    def process_file(self, input_path: Path, output_path: Path):
        """
        Process a file of clean sentences and generate corrupted pairs.

        Args:
            input_path: Path to input file with clean sentences
            output_path: Path to output JSONL file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:

            for line in f_in:
                sentence = line.strip()
                if not sentence:
                    continue

                corrupted, correct, edits = self.corrupt_sentence(sentence)

                data = {
                    'corrupted': corrupted,
                    'correct': correct,
                    'edits': edits,
                    'source': 'confusion_sets'
                }

                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Generate confusion set corruptions')
    parser.add_argument('--input', type=Path, required=True, help='Input file with clean sentences')
    parser.add_argument('--output', type=Path, required=True, help='Output JSONL file')
    parser.add_argument('--corruption-rate', type=float, default=0.15, help='Token corruption probability')

    args = parser.parse_args()

    corruptor = ConfusionSetCorruptor(corruption_rate=args.corruption_rate)
    corruptor.process_file(args.input, args.output)

    print(f"✓ Generated confusion set corruptions: {args.output}")


if __name__ == '__main__':
    main()
