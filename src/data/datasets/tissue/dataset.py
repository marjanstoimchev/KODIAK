"""
TissuePatchDataset - Dataset for multiplex tissue imaging patches.
Loads patches from CSV with magnification metadata.
"""

import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Optional, Union, List, Dict
from torch.utils.data import Dataset


def extract_label_from_filename(filepath: str) -> str:
    """
    Extract label from tissue patch filename pattern.

    Examples:
        coord_0_0_Bcell_state_None_lvl_2.tiff -> Bcell
        coord_5152_4704_EndothelialCells_state_None_lvl_2.tiff -> EndothelialCells
        CTL_CD8+KI67+_state_None_lvl_5.tiff -> CTL_CD8+KI67+

    Args:
        filepath: Path to image file

    Returns:
        Extracted label string
    """
    filename = Path(filepath).stem  # Remove extension

    # Find the label part: everything before "_state_"
    if '_state_' in filename:
        # Split by _state_ and take the part before
        before_state = filename.split('_state_')[0]

        # Remove coordinate prefix if present (coord_X_Y_)
        if before_state.startswith('coord_'):
            # Format: coord_0_0_LABEL
            parts = before_state.split('_')
            # Skip 'coord', x, y and take the rest
            if len(parts) > 3:
                label = '_'.join(parts[3:])
            else:
                label = before_state
        else:
            label = before_state
    else:
        # Fallback: use filename
        label = filename

    return label


class TissuePatchDataset(Dataset):
    """
    Dataset for loading multiplex tissue patches from CSV.

    CSV format:
        magnification,image_path
        10X,/path/to/patch1.tiff
        40X,/path/to/patch2.tiff

    Args:
        csv_file: Path to CSV file with columns [magnification, image_path]
        magnification_filter: Optional filter for specific magnification(s)
                            Can be string ('10X') or list (['10X', '40X'])
        root_path: Optional root directory to prepend to relative paths in CSV
                   If None, paths in CSV are used as-is.
                   Useful when running from different working directories (e.g., notebooks/).
    """

    def __init__(
        self,
        csv_file: str,
        magnification_filter: Optional[Union[str, List[str]]] = None,
        root_path: Optional[str] = None
    ):
        self.csv_file = Path(csv_file)
        self.root_path = Path(root_path) if root_path else None
        self.df = pd.read_csv(csv_file)

        # Filter by magnification if specified
        if magnification_filter:
            if isinstance(magnification_filter, str):
                self.df = self.df[self.df['magnification'] == magnification_filter]
            elif isinstance(magnification_filter, (list, tuple)):
                self.df = self.df[self.df['magnification'].isin(magnification_filter)]

            self.df = self.df.reset_index(drop=True)

        # Extract labels from filenames and create mappings
        self._extract_labels()

        print(f"TissuePatchDataset: Loaded {len(self.df)} patches from {csv_file}")
        if len(self.df) > 0:
            mags = self.df['magnification'].unique().tolist()
            print(f"  Available magnifications: {mags}")
            print(f"  Found {len(self.label_to_idx)} unique labels: {list(self.label_names.values())[:5]}{'...' if len(self.label_names) > 5 else ''}")

    def _extract_labels(self):
        """Extract labels from filenames and create label mappings."""
        all_labels = []
        for _, row in self.df.iterrows():
            label = extract_label_from_filename(row['image_path'])
            all_labels.append(label)

        # Create label mappings
        unique_labels = sorted(set(all_labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_names = {idx: label for label, idx in self.label_to_idx.items()}

        # Store labels for each sample
        self.labels = [self.label_to_idx[label] for label in all_labels]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns dict with image and metadata.
        Transforms will be applied by the DataModule.
        """
        # Convert numpy integers to Python int
        idx = int(idx)
        row = self.df.iloc[idx]
        img_path = Path(row['image_path'])

        # Prepend root_path if specified and path is relative
        if self.root_path is not None:
            if not img_path.is_absolute():
                img_path = self.root_path / img_path

        # Load image
        image = Image.open(img_path)

        # Convert to RGB (handles grayscale, RGBA, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Get label (integer index)
        label = self.labels[idx]

        return {
            'image': image,
            'label': label,  # Integer label index
            'magnification': row['magnification'],
            'image_path': str(img_path),
            'idx': idx,
        }

    def get_label_name(self, label_idx: int) -> str:
        """Get label name from integer index."""
        return self.label_names.get(label_idx, f"unknown_{label_idx}")

    def get_num_classes(self) -> int:
        """Get number of unique classes in the dataset."""
        return len(self.label_names)
