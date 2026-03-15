"""
Seq2Seq models for GEC: T5, mT5, mBART, BART.
"""

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    MBartForConditionalGeneration,
    BartForConditionalGeneration
)
from typing import Dict, Any


class GECSeq2SeqModel:
    """Wrapper for sequence-to-sequence GEC models."""

    MODEL_CLASSES = {
        't5': T5ForConditionalGeneration,
        'mt5': T5ForConditionalGeneration,
        'mbart': MBartForConditionalGeneration,
        'bart': BartForConditionalGeneration
    }

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize seq2seq GEC model.

        Args:
            model_name: HuggingFace model identifier (e.g., 'google/mt5-base')
            config: Model configuration dict
        """
        self.model_name = model_name
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def preprocess(self, text: str, prefix: str = "grammar: ") -> Dict:
        """
        Preprocess input text for the model.

        Args:
            text: Input corrupted sentence
            prefix: Task prefix (for T5-style models)

        Returns:
            Tokenized inputs
        """
        input_text = prefix + text
        return self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get('max_length', 512)
        )

    def generate(self, text: str, **generation_kwargs) -> str:
        """
        Generate correction for corrupted text.

        Args:
            text: Corrupted input sentence
            **generation_kwargs: Additional generation parameters

        Returns:
            Corrected sentence
        """
        inputs = self.preprocess(text)
        outputs = self.model.generate(
            **inputs,
            max_length=generation_kwargs.get('max_length', 512),
            num_beams=generation_kwargs.get('num_beams', 4),
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
