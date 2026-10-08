"""
HuggingFaceDataset - Dataset wrapper for Hugging Face Datasets.
Loads images and labels from HF Hub with support for streaming and splits.

Supports smart split detection for classification tasks:
- Detects available splits (train, test, val, validation)
- Returns split information for smart data loading
"""

import numpy as np
from PIL import Image
from typing import Optional, Dict, Any
from torch.utils.data import Dataset

try:
    from datasets import load_dataset, DatasetDict, concatenate_datasets, get_dataset_split_names
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def _hf_call(fn, *args, **kwargs):
    """Call a `datasets` function, passing ``trust_remote_code=True`` when supported.

    ``datasets`` < 4.0 needs ``trust_remote_code=True`` for script-based datasets,
    while ``datasets`` >= 4.0 removed the argument entirely. This keeps both working.
    """
    try:
        return fn(*args, trust_remote_code=True, **kwargs)
    except TypeError as e:
        if "trust_remote_code" not in str(e):
            raise
        return fn(*args, **kwargs)


def get_available_splits(dataset_name: str, cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Detect available splits in a HuggingFace dataset.

    Returns:
        Dict with keys 'train', 'val', 'test' and boolean values indicating availability.
        'val' is True if either 'val' or 'validation' split exists.
        'available' contains the original split names.
        'val_key' contains the actual key to use for validation split.
    """
    if not HF_AVAILABLE:
        raise ImportError("Hugging Face datasets not available. Install with: pip install datasets")

    try:
        splits = _hf_call(get_dataset_split_names, dataset_name)
    except Exception as e:
        print(f"Warning: Could not get split names for {dataset_name}: {e}")
        # Fallback: try loading and checking
        try:
            ds = _hf_call(load_dataset, dataset_name, cache_dir=cache_dir)
            if isinstance(ds, DatasetDict):
                splits = list(ds.keys())
            else:
                splits = ["train"]
        except Exception:
            splits = ["train"]

    # Normalize split names
    splits_lower = [s.lower() for s in splits]

    # Determine validation key (some datasets use 'val', others 'validation')
    val_key = None
    if 'validation' in splits_lower:
        val_key = 'validation'
    elif 'val' in splits_lower:
        val_key = 'val'

    return {
        'train': 'train' in splits_lower,
        'val': val_key is not None,
        'test': 'test' in splits_lower,
        'available': splits,
        'val_key': val_key,
    }


class HuggingFaceDataset(Dataset):
    """
    Generic dataset for loading images from Hugging Face Hub.

    Args:
        name: Dataset name on HF Hub (e.g., "timm/oxford-iiit-pet", "ILSVRC/imagenet-1k")
        split: Specific split to use. If None, concatenates available splits.
        streaming: Use streaming mode for large datasets.
        image_key: Column name containing images.
        label_key: Column name containing labels.
    """

    def __init__(
        self,
        name: str,
        split: Optional[str] = None,
        streaming: bool = False,
        image_key: str = "image",
        label_key: str = "label",
        cache_dir: Optional[str] = None,
    ):
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face datasets not available. Install with: pip install datasets")

        self.name = name
        self.streaming = streaming
        self.image_key = image_key
        self.label_key = label_key

        # Load Dataset
        print(f"HuggingFaceDataset: Loading {name}...")
        
        if split:
            self.dataset = _hf_call(
                load_dataset, name, split=split, streaming=streaming, cache_dir=cache_dir
            )
        else:
            # Load all splits
            dataset_dict = _hf_call(
                load_dataset, name, streaming=streaming, cache_dir=cache_dir
            )
            
            if isinstance(dataset_dict, DatasetDict):
                if streaming:
                    # Streaming doesn't support concatenation easily, take first split
                    first_split = list(dataset_dict.keys())[0]
                    self.dataset = dataset_dict[first_split]
                    print(f" Streaming mode: selected split '{first_split}'")
                else:
                    # Concatenate all splits
                    self.dataset = concatenate_datasets(list(dataset_dict.values()))
            else:
                self.dataset = dataset_dict

        # Resolve columns if defaults don't exist
        if not streaming:
            columns = self.dataset.column_names
            if self.image_key not in columns:
                for alt in ["img", "images", "picture", "photo", "file_name"]:
                    if alt in columns:
                        self.image_key = alt
                        break

            if self.label_key not in columns:
                for alt in ["labels", "fine_label", "class", "category", "target", "y"]:
                    if alt in columns:
                        self.label_key = alt
                        break

            print(f"HuggingFaceDataset: Loaded {len(self.dataset)} samples from {name}")
            print(f" Keys used - Image: '{self.image_key}', Label: '{self.label_key}'")

            # Build label encoding if labels are strings
            self._build_label_encoding()
        else:
            # For streaming mode, we'll build the encoding dynamically
            self.label_to_idx = {}
            self.label_names = {}

    def _build_label_encoding(self):
        """Build label-to-index mapping for string labels."""
        if self.label_key not in self.dataset.column_names:
            self.label_to_idx = {}
            self.label_names = {}
            print(" No label column found, skipping label encoding")
            return

        # Sample a few examples to check label type
        sample_label = self.dataset[0][self.label_key]

        # If labels are already numeric, no encoding needed
        if isinstance(sample_label, (int, np.integer)):
            self.label_to_idx = {}
            self.label_names = {}
            print(" Labels are already numeric, no encoding needed")
            return

        # Build encoding for string labels
        print(" Scanning dataset to build label encoding...")
        unique_labels = set()
        for item in self.dataset:
            label_val = item[self.label_key]
            if label_val is not None:
                unique_labels.add(label_val)

        # Create mappings
        sorted_labels = sorted(unique_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
        self.label_names = {idx: label for label, idx in self.label_to_idx.items()}

        print(f" Built label encoding with {len(self.label_to_idx)} classes:")
        print(f"   {sorted_labels}")

    def __len__(self) -> int:
        if self.streaming:
            return 1000000 # Arbitrary large number for streaming
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns dict with image and metadata.
        Transforms will be applied by the DataModule.
        """
        # Convert numpy integers to Python int (required by HuggingFace datasets)
        idx = int(idx)

        # Handle data retrieval (Streaming vs Random Access)
        sample = None
        if self.streaming:
            # Inefficient for random access, but necessary for streaming iterables
            for i, item in enumerate(self.dataset):
                if i == idx:
                    sample = item
                    break
            if sample is None:
                raise IndexError(f"Index {idx} out of range")
        else:
            sample = self.dataset[idx]

        # Extract and Process Image
        image_data = sample[self.image_key]
        
        # Handle various HF image formats (PIL, Bytes, Numpy)
        if isinstance(image_data, dict) and 'bytes' in image_data:
            from io import BytesIO
            image = Image.open(BytesIO(image_data['bytes']))
        elif isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        else:
            image = image_data # Assume already PIL or compatible

        # Ensure RGB
        if hasattr(image, 'mode') and image.mode != 'RGB':
            image = image.convert('RGB')

        # Extract Label
        label = -1
        if self.label_key and self.label_key in sample:
            raw_label = sample[self.label_key]

            # Handle string labels with encoding
            if isinstance(raw_label, str):
                if self.label_to_idx:
                    # Use pre-built encoding
                    label = self.label_to_idx.get(raw_label, -1)
                else:
                    # Build encoding dynamically for streaming mode
                    if raw_label not in self.label_to_idx:
                        new_idx = len(self.label_to_idx)
                        self.label_to_idx[raw_label] = new_idx
                        self.label_names[new_idx] = raw_label
                    label = self.label_to_idx[raw_label]
            else:
                # Already numeric
                label = raw_label

        # Try to find a name/ID, otherwise generate one
        image_name = str(sample.get('id', sample.get('filename', f"{self.name}_{idx}")))

        return {
            'image': image,
            'label': label,
            'image_name': image_name,
            'idx': idx,
            'dataset_name': self.name
        }