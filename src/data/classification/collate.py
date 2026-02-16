# =============================================================================
# Classification Collate Function
# =============================================================================
# Simple collate function for classification (no masking needed).
# =============================================================================

import torch


def classification_collate(batch):
    """
    Simple collate function for classification (no masking needed).

    Just stacks images and labels - that's it.
    """
    images = torch.stack([x["images"] for x in batch])
    labels = torch.tensor([x.get("label", -1) for x in batch])
    return {"images": images, "labels": labels}
