"""
Random character/token operations for cheap synthetic data augmentation.

Insertions, deletions, swaps at character or token level.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Tuple, List, Dict


class RandomOpsCorruptor:
    """Generate corrupted sentences using random character/token operations."""

    def __init__(self, corruption_rate: float = 0.1):
        self.corruption_rate = corruption_rate

    def corrupt_token_char_level(self, token: str) -> Tuple[str, str]:
        """Apply random character-level operations to a token."""
        if len(token) < 2:
            return token, 'none'

        operation = random.choice(['insert', 'delete', 'swap'])

        if operation == 'insert':
            pos = random.randint(0, len(token))
            char = random.choice('abcdefghijklmnopqrstuvwxyzäö')
            return token[:pos] + char + token[pos:], 'char_insert'

        elif operation == 'delete':
            pos = random.randint(0, len(token) - 1)
            return token[:pos] + token[pos+1:], 'char_delete'

        else:  # swap
            pos = random.randint(0, len(token) - 2)
            chars = list(token)
            chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
            return ''.join(chars), 'char_swap'

    def corrupt_sentence(self, sentence: str) -> Tuple[str, str, List[Dict]]:
        """Corrupt a sentence with random operations."""
        tokens = sentence.split()
        corrupted_tokens = []
        edits = []

        for i, token in enumerate(tokens):
            if random.random() < self.corruption_rate:
                corrupted, op_type = self.corrupt_token_char_level(token)
                corrupted_tokens.append(corrupted)
                edits.append({
                    'position': i,
                    'original': token,
                    'corrupted': corrupted,
                    'type': op_type
                })
            else:
                corrupted_tokens.append(token)

        return ' '.join(corrupted_tokens), sentence, edits

    def process_file(self, input_path: Path, output_path: Path):
        """Process input file and generate corrupted pairs."""
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
                    'source': 'random_ops'
                }

                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Generate random operation corruptions')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--corruption-rate', type=float, default=0.1)

    args = parser.parse_args()

    corruptor = RandomOpsCorruptor(corruption_rate=args.corruption_rate)
    corruptor.process_file(args.input, args.output)

    print(f"✓ Generated random ops corruptions: {args.output}")


if __name__ == '__main__':
    main()
