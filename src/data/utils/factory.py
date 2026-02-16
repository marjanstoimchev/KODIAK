"""
Dataset Factory with Registry Pattern

This module implements a flexible dataset factory using the registry pattern,
allowing easy extension without modifying core code (Open/Closed Principle).
"""

from typing import Dict, Type, Any, Optional
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """
    Registry for dataset classes following the Open/Closed Principle.

    This allows adding new datasets without modifying the factory code.
    Simply register new dataset classes with a decorator.

    Example:
        @DatasetRegistry.register("my_dataset")
        class MyDataset(Dataset):
            def __init__(self, **kwargs):
                ...

        # Later, create instance
        dataset = DatasetRegistry.create("my_dataset", **config)
    """

    _registry: Dict[str, Type[Dataset]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a dataset class.

        Args:
            name: Unique identifier for this dataset type

        Example:
            @DatasetRegistry.register("pancreatic")
            class PancreaticDataset(Dataset):
                pass
        """
        def decorator(dataset_cls: Type[Dataset]) -> Type[Dataset]:
            if name in cls._registry:
                logger.warning(
                    f"Dataset '{name}' is already registered. "
                    f"Overwriting with {dataset_cls.__name__}"
                )

            cls._registry[name] = dataset_cls
            logger.debug(f"Registered dataset: '{name}' -> {dataset_cls.__name__}")
            return dataset_cls

        return decorator

    @classmethod
    def create(cls, dataset_type: str, **kwargs) -> Dataset:
        """
        Create a dataset instance by type.

        Args:
            dataset_type: Registered dataset type (e.g., 'pancreatic', 'huggingface')
            **kwargs: Arguments to pass to the dataset constructor

        Returns:
            Dataset instance

        Raises:
            ValueError: If dataset type is not registered

        Example:
            dataset = DatasetRegistry.create(
                "pancreatic",
                root_dir="/path/to/dataset"
            )
        """
        if dataset_type not in cls._registry:
            available = list(cls._registry.keys())

            # Provide helpful error message with suggestions
            error_msg = (
                f"ERROR: Unknown dataset type: '{dataset_type}'\n"
                f"\n"
                f"Available dataset types:\n"
            )

            for name in available:
                dataset_cls = cls._registry[name]
                error_msg += f"  - '{name}' ({dataset_cls.__name__})\n"

            error_msg += (
                f"\n"
                f"How to register a new dataset:\n"
                f"  1. Import DatasetRegistry: from data.factory import DatasetRegistry\n"
                f"  2. Decorate your class: @DatasetRegistry.register('your_name')\n"
                f"  3. Your dataset will be available via the factory\n"
            )

            raise ValueError(error_msg)

        dataset_cls = cls._registry[dataset_type]

        try:
            return dataset_cls(**kwargs)
        except TypeError as e:
            # Provide helpful error about missing/invalid arguments
            raise TypeError(
                f"ERROR: Failed to create dataset '{dataset_type}' ({dataset_cls.__name__})\n"
                f"\n"
                f"Error: {str(e)}\n"
                f"\n"
                f"Suggestion: Check that you're providing the correct arguments for {dataset_cls.__name__}.\n"
                f"See the dataset class documentation for required parameters."
            ) from e

    @classmethod
    def list_datasets(cls) -> Dict[str, Type[Dataset]]:
        """
        List all registered datasets.

        Returns:
            Dictionary mapping dataset names to their classes
        """
        return cls._registry.copy()

    @classmethod
    def is_registered(cls, dataset_type: str) -> bool:
        """Check if a dataset type is registered."""
        return dataset_type in cls._registry


# Auto-register built-in datasets
def _register_builtin_datasets():
    """Auto-register built-in dataset types."""
    try:
        from src.data.datasets.custom.dataset import CustomPatchDataset
        DatasetRegistry.register("pancreatic")(CustomPatchDataset)
    except ImportError as e:
        logger.warning(f"Could not register pancreatic dataset: {e}")

    try:
        from src.data.datasets.huggingface.dataset import HuggingFaceDataset
        DatasetRegistry.register("huggingface")(HuggingFaceDataset)
    except ImportError as e:
        logger.warning(f"Could not register HuggingFace dataset: {e}")


# Register on module import
_register_builtin_datasets()
