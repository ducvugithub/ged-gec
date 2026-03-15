"""
GECToR-style token classification model for GEC.

Per-token classification over edit operations: {KEEP, DELETE, REPLACE→X, INSERT→X}
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple


class GECToRModel(nn.Module):
    """Token classification model for grammatical error correction."""

    def __init__(self, encoder_name: str, num_labels: int, num_detect_labels: int = 2):
        """
        Initialize GECToR model.

        Args:
            encoder_name: HuggingFace encoder model (e.g., 'xlm-roberta-base')
            num_labels: Number of correction labels (edit vocabulary size)
            num_detect_labels: Number of error detection labels (default: binary)
        """
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size

        # Error detection head (binary: correct/incorrect)
        self.detect_head = nn.Linear(hidden_size, num_detect_labels)

        # Correction head (edit operations)
        self.correct_head = nn.Linear(hidden_size, num_labels)

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None, labels=None, detect_labels=None):
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Correction labels (optional, for training)
            detect_labels: Detection labels (optional, for training)

        Returns:
            Dict with logits and optional loss
        """
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # Prediction heads
        detect_logits = self.detect_head(sequence_output)
        correct_logits = self.correct_head(sequence_output)

        loss = None
        if labels is not None and detect_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            detect_loss = loss_fct(detect_logits.view(-1, 2), detect_labels.view(-1))
            correct_loss = loss_fct(correct_logits.view(-1, self.num_labels), labels.view(-1))
            loss = detect_loss + correct_loss

        return {
            'loss': loss,
            'detect_logits': detect_logits,
            'correct_logits': correct_logits
        }

    @property
    def num_labels(self):
        return self.correct_head.out_features


class GECToRInference:
    """Iterative inference for GECToR model."""

    def __init__(self, model, tokenizer, max_iterations: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.max_iterations = max_iterations
        self.model.eval()

    def predict(self, text: str) -> str:
        """
        Iteratively correct text until no more edits or max iterations.

        Args:
            text: Input corrupted text

        Returns:
            Corrected text
        """
        current_text = text

        for i in range(self.max_iterations):
            inputs = self.tokenizer(current_text, return_tensors='pt')

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get predictions
            detect_preds = outputs['detect_logits'].argmax(dim=-1)
            correct_preds = outputs['correct_logits'].argmax(dim=-1)

            # Apply edits
            corrected_text = self._apply_edits(current_text, detect_preds, correct_preds)

            # Stop if no changes
            if corrected_text == current_text:
                break

            current_text = corrected_text

        return current_text

    def _apply_edits(self, text: str, detect_preds, correct_preds) -> str:
        """Apply predicted edits to text."""
        # TODO: Implement edit application logic
        # This requires mapping from edit labels to actual operations
        return text
