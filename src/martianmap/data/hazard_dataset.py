"""Ortho (1 m) -> hazard label tiles for the U-Net, read as windows from the GeoTIFFs."""

from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from martianmap.terrain.hazard_labels import IGNORE

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # ImageNet; grayscale is repeated to 3 channels
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_input(img01: np.ndarray) -> torch.Tensor:
    """(H, W) image in [0, 1] -> normalised (3, H, W) float tensor."""
    x = torch.from_numpy(np.ascontiguousarray(img01, dtype=np.float32))[None].expand(3, -1, -1)
    return (x - MEAN) / STD


def augment(img01: np.ndarray, lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Random rot90 + flip (slope magnitude is rotation invariant) and brightness/contrast/gamma (image only).

    Uses torch RNG so DataLoader workers get different streams.
    """
    k = int(torch.randint(4, ()))
    img01, lab = np.rot90(img01, k), np.rot90(lab, k)
    if torch.rand(()) < 0.5:
        img01, lab = img01[:, ::-1], lab[:, ::-1]
    contrast, bright, gamma = (float(torch.empty(()).uniform_(a, b)) for a, b in ((0.8, 1.2), (-0.1, 0.1), (0.8, 1.25)))
    img01 = np.clip((img01 - 0.5) * contrast + 0.5 + bright, 0, 1) ** gamma
    return img01, lab


class HazardTiles(Dataset):
    """Tiles at pixel origins (row, col); labels set to IGNORE where the ortho has no data (0)."""

    def __init__(self, ortho: Path, labels: Path, origins: list[tuple[int, int]], tile: int, train: bool) -> None:
        self.ortho, self.labels, self.origins, self.tile, self.train = ortho, labels, origins, tile, train
        self._src = None  # opened lazily so each DataLoader worker gets its own handles

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._src is None:
            self._src = rasterio.open(self.ortho), rasterio.open(self.labels)
        r, c = self.origins[i]
        win = Window(c, r, self.tile, self.tile)
        img, lab = self._src[0].read(1, window=win), self._src[1].read(1, window=win)
        lab[img == 0] = IGNORE
        img01 = img.astype(np.float32) / 255
        if self.train:
            img01, lab = augment(img01, lab)
        return to_input(img01), torch.from_numpy(np.ascontiguousarray(lab, dtype=np.int64))
