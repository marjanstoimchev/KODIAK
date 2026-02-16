"""
CustomPatchDataset - Dataset for image patches from folder or CSV.

Supports two loading modes:
  1. Folder-based (ImageFolder style): root_dir with class subfolders
  2. CSV-based: CSV file with image_path and optional label columns
"""

import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Optional, Union, List
from torch.utils.data import Dataset


class CustomPatchDataset(Dataset):
    """
    Dataset for loading image patches from class folders or CSV.

    Folder mode (recommended) — supports both flat and nested structures:

      Flat (single slide):
        root_dir/
            Bcell/
                image1.png
            Tcell/
                image2.png

      Nested (multiple slides):
        root_dir/
            SLIDE-3210/
                Bcell/
                    image1.png
                Tcell/
                    image2.png
            SLIDE-4567/
                Bcell/
                    image3.png

    CSV mode:
        image_path,label
        /path/to/patch1.png,Bcell

    Args:
        root_dir: Path to directory with class subfolders, or parent of slide dirs
        csv_file: Path to CSV file (alternative to root_dir)
        magnification_filter: Optional filter for specific magnification(s) (CSV mode only)
        root_path: Optional root directory to prepend to relative paths in CSV
    """

    EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}

    def __init__(
        self,
        root_dir: Optional[str] = None,
        csv_file: Optional[str] = None,
        magnification_filter: Optional[Union[str, List[str]]] = None,
        root_path: Optional[str] = None,
    ):
        if root_dir is not None:
            self._load_from_folder(root_dir)
        elif csv_file is not None:
            self._load_from_csv(csv_file, magnification_filter, root_path)
        else:
            raise ValueError("Either root_dir or csv_file must be provided")

        # Build label mappings
        unique_labels = sorted(set(self.all_labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_names = {idx: label for label, idx in self.label_to_idx.items()}
        self.labels = [self.label_to_idx[label] for label in self.all_labels]

        print(f"CustomPatchDataset: Loaded {len(self.image_paths)} images, "
              f"{len(self.label_to_idx)} classes: {list(self.label_to_idx.keys())}")

    def _load_from_folder(self, root_dir: str):
        """Load dataset from directory with class subfolders.

        Auto-detects structure:
          - Flat: root_dir/ClassName/image.png
          - Nested: root_dir/SlideID/ClassName/image.png
        """
        root = Path(root_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        self.image_paths = []
        self.all_labels = []

        subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if not subdirs:
            raise RuntimeError(f"No subdirectories found in {root_dir}")

        # Check first subdir: images inside → flat, only dirs inside → nested
        first_has_images = any(
            f.suffix.lower() in self.EXTENSIONS
            for f in subdirs[0].iterdir() if f.is_file()
        )

        if first_has_images:
            # Flat: root_dir/ClassName/image.png
            self._scan_class_dirs(root)
        else:
            # Nested: root_dir/SlideID/ClassName/image.png
            for slide_dir in subdirs:
                self._scan_class_dirs(slide_dir)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root_dir}. "
                             f"Expected class subfolders with image files.")

    def _scan_class_dirs(self, parent: Path):
        """Scan class subdirectories under parent and collect images."""
        for class_dir in sorted(parent.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in self.EXTENSIONS:
                    self.image_paths.append(str(img_file))
                    self.all_labels.append(label)

    def _load_from_csv(self, csv_file: str, magnification_filter, root_path):
        """Load dataset from CSV file."""
        self.root_path = Path(root_path) if root_path else None
        df = pd.read_csv(csv_file)

        # Filter by magnification if specified
        if magnification_filter and 'magnification' in df.columns:
            if isinstance(magnification_filter, str):
                df = df[df['magnification'] == magnification_filter]
            elif isinstance(magnification_filter, (list, tuple)):
                df = df[df['magnification'].isin(magnification_filter)]
            df = df.reset_index(drop=True)

        self.image_paths = []
        self.all_labels = []

        for _, row in df.iterrows():
            img_path = row['image_path']
            if self.root_path and not Path(img_path).is_absolute():
                img_path = str(self.root_path / img_path)
            self.image_paths.append(img_path)

            # Resolve label: CSV column > parent folder > filename
            if 'label' in df.columns and pd.notna(row.get('label')):
                self.all_labels.append(str(row['label']))
            else:
                parent = Path(row['image_path']).parent.name
                self.all_labels.append(parent if parent else Path(row['image_path']).stem)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        idx = int(idx)
        img_path = self.image_paths[idx]

        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return {
            'image': image,
            'label': self.labels[idx],
            'image_path': img_path,
            'idx': idx,
        }

    def get_label_name(self, label_idx: int) -> str:
        return self.label_names.get(label_idx, f"unknown_{label_idx}")

    def get_num_classes(self) -> int:
        return len(self.label_names)
