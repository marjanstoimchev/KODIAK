"""
Simple Configuration Validation

Validates critical configuration fields without heavy dependencies.
"""

from pathlib import Path
from typing import Any, Dict, List


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration and return list of errors.

    Args:
        config: Configuration dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Experiment validation
    if 'experiment' in config:
        exp = config['experiment']
        if not exp.get('name') or exp['name'].strip() == "":
            errors.append("experiment.name cannot be empty")

        # Check for invalid filesystem characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in exp.get('name', '') for char in invalid_chars):
            errors.append(f"experiment.name contains invalid characters: {exp['name']}")

    # Data validation
    if 'data' in config:
        data = config['data']
        dataset_type = data.get('dataset_type')

        # Dataset-specific requirements
        if dataset_type == 'tissue':
            csv_path = data.get('csv_path')
            if not csv_path:
                errors.append("data.csv_path is required for dataset_type='tissue'")
            elif not Path(csv_path).exists():
                errors.append(f"CSV file not found: {csv_path}")

        elif dataset_type == 'huggingface':
            if not data.get('hf_dataset_name'):
                errors.append("data.hf_dataset_name is required for dataset_type='huggingface'")

        # Split validation
        train_split = data.get('train_split', 0.9)
        val_split = data.get('val_split', 0.1)
        if train_split + val_split > 1.0:
            errors.append(
                f"train_split ({train_split}) + val_split ({val_split}) must be <= 1.0"
            )

        # Batch size sanity check
        batch_size = data.get('batch_size', 32)
        if batch_size < 1 or batch_size > 4096:
            errors.append(f"batch_size ({batch_size}) should be between 1 and 4096")

    # Model validation
    if 'model' in config:
        model = config['model']
        img_size = model.get('image_size', 256)
        patch_size = model.get('vit_patch_size', 16)

        if img_size % patch_size != 0:
            errors.append(
                f"image_size ({img_size}) must be divisible by vit_patch_size ({patch_size})"
            )

        embed_dim = model.get('vit_embed_dim', 384)
        num_heads = model.get('vit_heads', 6)

        if embed_dim % num_heads != 0:
            errors.append(
                f"vit_embed_dim ({embed_dim}) must be divisible by vit_heads ({num_heads})"
            )

        # Mask strategy validation
        mask_strategy = model.get('mask_strategy', 'dinov3')
        if mask_strategy != 'dinov3':
            errors.append(
                f"Only 'dinov3' mask_strategy is supported, got '{mask_strategy}'"
            )

    return errors


def check_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration and print errors.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    errors = validate_config(config)

    if errors:
        print("\n" + "=" * 80)
        print("CONFIGURATION ERRORS:")
        print("=" * 80)
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
        print("=" * 80 + "\n")
        return False

    return True
