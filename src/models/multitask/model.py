"""
Multitask GED + GEC joint model.

Shared encoder with two heads:
- GED head: token-level error detection
- GEC head: seq2seq correction
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSeq2SeqLM
from typing import Dict, Optional


class MultitaskGEDGECModel(nn.Module):
    """Joint error detection and correction model."""

    def __init__(self, encoder_name: str, decoder_name: str, num_detect_labels: int = 3):
        """
        Initialize multitask model.

        Args:
            encoder_name: Encoder model (e.g., 'xlm-roberta-base')
            decoder_name: Decoder model (e.g., 'google/mt5-base')
            num_detect_labels: Number of detection labels (e.g., 3: correct/error-type-1/error-type-2)
        """
        super().__init__()

        # Shared encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)

        # GED head (error detection)
        hidden_size = self.encoder.config.hidden_size
        self.ged_head = nn.Linear(hidden_size, num_detect_labels)

        # GEC head (correction decoder)
        self.gec_decoder = AutoModelForSeq2SeqLM.from_pretrained(decoder_name)

        # Project encoder hidden states to decoder input
        decoder_hidden_size = self.gec_decoder.config.d_model
        self.encoder_to_decoder = nn.Linear(hidden_size, decoder_hidden_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        ged_labels=None,
        gec_labels=None
    ):
        """
        Forward pass with joint training.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            decoder_input_ids: Decoder input IDs (for GEC)
            decoder_attention_mask: Decoder attention mask
            ged_labels: Error detection labels
            gec_labels: Correction labels

        Returns:
            Dict with losses and logits
        """
        # Encode input
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        encoder_hidden = encoder_outputs.last_hidden_state

        # GED head
        ged_logits = self.ged_head(encoder_hidden)

        # GEC head
        decoder_hidden = self.encoder_to_decoder(encoder_hidden)

        gec_outputs = self.gec_decoder(
            encoder_outputs=(decoder_hidden,),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=gec_labels
        )

        # Compute losses
        total_loss = None
        ged_loss = None
        gec_loss = gec_outputs.loss

        if ged_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            ged_loss = loss_fct(ged_logits.view(-1, self.num_detect_labels), ged_labels.view(-1))

            if gec_loss is not None:
                total_loss = ged_loss + gec_loss
            else:
                total_loss = ged_loss

        return {
            'loss': total_loss,
            'ged_loss': ged_loss,
            'gec_loss': gec_loss,
            'ged_logits': ged_logits,
            'gec_logits': gec_outputs.logits
        }

    @property
    def num_detect_labels(self):
        return self.ged_head.out_features
