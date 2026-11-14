"""Image-only slope hazard U-Net (3 classes: safe / caution / hazard)."""

from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

from martianmap.data.hazard_dataset import to_input


def build_model(encoder_weights: str | None = "imagenet") -> torch.nn.Module:
    """smp U-Net, ResNet-34 encoder, 3-channel input (grayscale repeated), 3 classes."""
    return smp.Unet("resnet34", encoder_weights=encoder_weights, in_channels=3, classes=3)


def load_model(ckpt: Path, device: str = "cpu") -> torch.nn.Module:
    """Model in eval mode from a checkpoint written by train/train_hazard.py."""
    model = build_model(encoder_weights=None)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    return model.eval().to(device)


def window_starts(n: int, tile: int, stride: int) -> list[int]:
    """Window starts covering [0, n) with the last window flush to the end. Requires n >= tile."""
    starts = list(range(0, n - tile + 1, stride))
    return starts if starts[-1] == n - tile else starts + [n - tile]


@torch.no_grad()
def predict_probs(model: torch.nn.Module, img01: np.ndarray, tile: int = 512, stride: int = 256,
                  device: str = "cuda", batch: int = 8) -> np.ndarray:
    """(3, H, W) class probabilities for a large (H, W) image in [0, 1]: overlapping windows,
    blended with a Hann-like weight so window edges (little context) count least."""
    h, w = img01.shape
    assert h >= tile and w >= tile, "image smaller than one tile"
    ramp = np.hanning(tile + 2)[1:-1].astype(np.float32)
    weight = torch.from_numpy(np.outer(ramp, ramp)).to(device)
    probs = torch.zeros(3, h, w, device=device)
    norm = torch.zeros(h, w, device=device)
    coords = [(r, c) for r in window_starts(h, tile, stride) for c in window_starts(w, tile, stride)]
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        x = torch.stack([to_input(img01[r:r + tile, c:c + tile]) for r, c in chunk]).to(device)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=device != "cpu"):
            p = model(x).float().softmax(1)
        for (r, c), pi in zip(chunk, p):
            probs[:, r:r + tile, c:c + tile] += pi * weight
            norm[r:r + tile, c:c + tile] += weight
    return (probs / norm).cpu().numpy()
