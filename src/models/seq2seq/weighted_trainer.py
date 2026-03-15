"""
Weighted Seq2Seq Trainer for balanced GEC training.

Uses training_weight from augmented data to ensure equal contribution
from all raw examples regardless of augmentation count.
"""

import torch
from transformers import Seq2SeqTrainer
from typing import Dict, Optional


class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom Seq2SeqTrainer that applies training weights for balanced learning.

    The training_weight field ensures that raw examples with many augmented
    samples don't dominate training over those with fewer samples.
    """

    def __init__(self, *args, use_weights: bool = True, **kwargs):
        """
        Initialize weighted trainer.

        Args:
            use_weights: Whether to apply training weights (default: True)
            *args, **kwargs: Passed to Seq2SeqTrainer
        """
        super().__init__(*args, **kwargs)
        self.use_weights = use_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute weighted loss for balanced training.

        Args:
            model: The model
            inputs: Input batch (may contain 'training_weight')
            return_outputs: Whether to return outputs

        Returns:
            Weighted loss (and outputs if return_outputs=True)
        """
        # Extract and remove training_weight from inputs
        # (model doesn't expect this field)
        training_weight = inputs.pop('training_weight', None)

        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        # Apply weights if available and enabled
        if self.use_weights and training_weight is not None:
            # Ensure weights are on same device as loss
            if isinstance(training_weight, (list, tuple)):
                training_weight = torch.tensor(
                    training_weight,
                    dtype=loss.dtype,
                    device=loss.device
                )
            elif not isinstance(training_weight, torch.Tensor):
                training_weight = torch.tensor(
                    [training_weight],
                    dtype=loss.dtype,
                    device=loss.device
                )
            else:
                training_weight = training_weight.to(
                    dtype=loss.dtype,
                    device=loss.device
                )

            # Apply weights
            # Note: HF Trainer averages loss across batch, so we apply weights then average
            if loss.dim() == 0:  # Scalar loss (already reduced)
                loss = loss * training_weight.mean()
            else:  # Per-sample loss
                loss = (loss * training_weight).mean()

        return (loss, outputs) if return_outputs else loss
