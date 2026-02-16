"""
Create DataFrame from sampled patches with magnification and image_path columns.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def create_patches_dataframe(
    sampled_root: str,
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Create DataFrame from sampled patches directory.

    Args:
        sampled_root: Root directory containing sampled patches (e.g., 'DATASET-1_SAMPLED')
        output_csv: Optional path to save CSV file

    Returns:
        DataFrame with columns: magnification, image_path
    """
    sampled_path = Path(sampled_root)

    if not sampled_path.exists():
        raise ValueError(f"Sampled directory not found: {sampled_root}")

    # Collect all .tiff files
    patches = []

    for tiff_file in sampled_path.rglob("*.tiff"):
        # Extract magnification from directory structure
        # Structure: SAMPLED_ROOT/SAMPLE_NAME/MAGNIFICATION/patches.tiff
        parts = tiff_file.parts

        # Find magnification (10X, 40X, etc.)
        mag = None
        for part in parts:
            if part.endswith('X'):
                mag = part
                break

        if mag is None:
            mag = 'unknown'

        patches.append({
            'magnification': mag,
            'image_path': str(tiff_file.absolute())
        })

    # Create DataFrame
    df = pd.DataFrame(patches)

    # Sort by path for consistency
    df = df.sort_values('image_path').reset_index(drop=True)

    # Save if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved DataFrame to: {output_csv}")

    print(f"Total patches: {len(df)}")
    print(f"Magnifications: {df['magnification'].unique().tolist()}")

    return df


if __name__ == '__main__':
    # Example usage
    df = create_patches_dataframe(
        sampled_root='DATASET-1_SAMPLED',
        output_csv='../patches.csv'
    )

    print("\nDataFrame preview:")
    print(df.head())
    print("\nMagnification distribution:")
    print(df['magnification'].value_counts())
