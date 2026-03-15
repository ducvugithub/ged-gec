"""
Trainer for GECToR-style token classification models.
"""

from typing import Dict, Any
from pathlib import Path


def train(config: Dict[str, Any]):
    """
    Train a GECToR token classification model.

    Args:
        config: Configuration dictionary
    """
    print(f"Training GECToR model with config: {config['model']}")

    # TODO: Implement GECToR training
    # Key steps:
    # 1. Load and preprocess data (convert to edit labels)
    # 2. Build edit vocabulary from training data
    # 3. Initialize model with encoder + classification heads
    # 4. Train with custom training loop or HF Trainer
    # 5. Implement iterative inference

    raise NotImplementedError("GECToR training not yet implemented")


if __name__ == '__main__':
    import yaml
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            config = yaml.safe_load(f)
        train(config)
