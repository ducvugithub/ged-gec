"""
Main training dispatcher — reads config and calls model-specific trainer.
"""

import argparse
from pathlib import Path
import yaml
from typing import Dict, Any


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_model(config: Dict[str, Any]):
    """
    Dispatch to appropriate model trainer based on config.

    Args:
        config: Configuration dictionary with 'model_type' key
    """
    model_type = config['model']['type']

    if model_type in ['t5', 'mt5', 'mbart', 'bart']:
        from src.models.seq2seq.trainer import train as train_seq2seq
        train_seq2seq(config)

    elif model_type == 'gector':
        from src.models.gector.trainer import train as train_gector
        train_gector(config)

    elif model_type == 'copy':
        from src.models.copy.trainer import train as train_copy
        train_copy(config)

    elif model_type == 'llm':
        from src.models.llm.trainer import train as train_llm
        train_llm(config)

    elif model_type == 'multitask':
        from src.models.multitask.trainer import train as train_multitask
        train_multitask(config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description='Train a GEC model')
    parser.add_argument('--config', type=Path, required=True, help='Path to config YAML file')

    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_config(args.config)
    train_model(config)


if __name__ == '__main__':
    main()
