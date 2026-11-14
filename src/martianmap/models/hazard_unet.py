"""Image-only slope hazard U-Net (3 classes: safe / caution / hazard)."""

from pathlib import Path

import segmentation_models_pytorch as smp
import torch


def build_model(encoder_weights: str | None = "imagenet") -> torch.nn.Module:
    """smp U-Net, ResNet-34 encoder, 3-channel input (grayscale repeated), 3 classes."""
    return smp.Unet("resnet34", encoder_weights=encoder_weights, in_channels=3, classes=3)


def load_model(ckpt: Path, device: str = "cpu") -> torch.nn.Module:
    """Model in eval mode from a checkpoint written by train/train_hazard.py."""
    model = build_model(encoder_weights=None)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    return model.eval().to(device)
