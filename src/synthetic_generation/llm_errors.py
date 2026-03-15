"""
LLM-generated errors for high-quality synthetic data.

Prompts an LLM to introduce realistic L1-specific learner errors.
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import os


CORRUPTION_PROMPT = """You are a Finnish language teacher creating training data for a grammatical error correction system.

Given a grammatically correct Finnish sentence, introduce 1-3 realistic learner errors that native English speakers commonly make when learning Finnish. Focus on:
- Morphological errors (case, tense, agreement)
- Word order issues
- Vocabulary confusion
- Missing or incorrect particles

Return ONLY the corrupted sentence, nothing else.

Correct sentence: {sentence}
Corrupted sentence:"""


class LLMCorruptor:
    """Generate corrupted sentences using LLM prompting."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize LLM corruptor.

        Args:
            model_name: OpenAI model name or local model path
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # TODO: Initialize OpenAI client or local model
        # from openai import OpenAI
        # self.client = OpenAI(api_key=self.api_key)

    def corrupt_sentence(self, sentence: str) -> str:
        """
        Corrupt a sentence using LLM generation.

        Args:
            sentence: Clean input sentence

        Returns:
            Corrupted sentence
        """
        # TODO: Implement LLM call
        # prompt = CORRUPTION_PROMPT.format(sentence=sentence)
        # response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.7
        # )
        # return response.choices[0].message.content.strip()
        return sentence

    def process_file(self, input_path: Path, output_path: Path, max_examples: Optional[int] = None):
        """Process input file and generate LLM corruptions."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:

            for i, line in enumerate(f_in):
                if max_examples and i >= max_examples:
                    break

                sentence = line.strip()
                if not sentence:
                    continue

                corrupted = self.corrupt_sentence(sentence)

                if corrupted != sentence:
                    data = {
                        'corrupted': corrupted,
                        'correct': sentence,
                        'source': f'llm_{self.model_name}'
                    }
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} sentences...")


def main():
    parser = argparse.ArgumentParser(description='Generate LLM-based corruptions')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--max-examples', type=int, help='Limit number of examples (for cost control)')

    args = parser.parse_args()

    corruptor = LLMCorruptor(model_name=args.model)
    corruptor.process_file(args.input, args.output, max_examples=args.max_examples)

    print(f"✓ Generated LLM corruptions: {args.output}")


if __name__ == '__main__':
    main()
