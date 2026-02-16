# =============================================================================
# Classification Transforms
# =============================================================================
# Lightweight transforms for classification training and validation.
# Converts tensors to PIL, applies augmentations, converts back.
# =============================================================================

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class ClassificationTrainTransform:
    """
    Classification training transforms - EXACT match to DinoV3LightningTraining.
    Converts tensor to PIL, applies transforms, converts back.

    Pipeline (EXACT DinoV3):
    - Resize to (224, 224) with BICUBIC
    - RandomHorizontalFlip (p=0.5)
    - RandomVerticalFlip (p=0.1)
    - RandomRotation (degrees=15)
    - ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    - RandomAffine (degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
    - ToTensor
    - Normalize (ImageNet stats)
    """
    def __init__(self, image_size: int = 256, mean: list = None, std: list = None):
        mean = mean or [0.485, 0.456, 0.406]
        std = std or [0.229, 0.224, 0.225]
        self.image_size = image_size
        self.mean = mean
        self.std = std

        # PIL-based transforms - EXACT match to DinoV3LightningTraining
        self.pil_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (C, H, W) tensor in [0,1]
        Returns: (3, H', W') normalized tensor
        """
        # Handle grayscale images
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(3, 1, 1)
        elif x.dim() == 4:
            x = x[0]

        # Ensure 3 channels
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        elif x.size(0) > 3:
            x = x[:3]

        # Convert tensor to PIL, apply transforms
        pil_img = TF.to_pil_image(x)
        return self.pil_transform(pil_img)


class ClassificationValTransform:
    """
    Classification validation transforms - EXACT match to DinoV3LightningTraining.
    Converts tensor to PIL, applies transforms, converts back.

    Pipeline (EXACT DinoV3):
    - Resize to (256, 256) with BICUBIC
    - CenterCrop to 256
    - ToTensor
    - Normalize (ImageNet stats)
    """
    def __init__(self, image_size: int = 256, mean: list = None, std: list = None):
        mean = mean or [0.485, 0.456, 0.406]
        std = std or [0.229, 0.224, 0.225]

        # PIL-based transforms - EXACT match to DinoV3LightningTraining
        self.pil_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (C, H, W) tensor in [0,1]
        Returns: (3, H', W') normalized tensor
        """
        # Handle grayscale images
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(3, 1, 1)
        elif x.dim() == 4:
            x = x[0]

        # Ensure 3 channels
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        elif x.size(0) > 3:
            x = x[:3]

        # Convert tensor to PIL, apply transforms
        pil_img = TF.to_pil_image(x)
        return self.pil_transform(pil_img)
