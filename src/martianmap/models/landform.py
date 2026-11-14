"""DoMars16k landform classifier (pretrained torchvision ResNet) + sliding-window landform map."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from rasterio.transform import Affine

# ImageFolder order (alphabetical); verified: pretrained resnet50 gets 93.1% on the DoMars16k test split
CLASSES = ("aec", "ael", "cli", "cra", "fse", "fsf", "fsg", "fss", "mix", "rid", "rou", "sfe", "sfx", "smo", "tex")
UNKNOWN = 255
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_domars(weights: Path, arch: str = "resnet50", device: str = "cpu") -> torch.nn.Module:
    """Pretrained DoMars16k network from models.zip (state dict keys are prefixed with 'net.')."""
    net = getattr(torchvision.models, arch)(num_classes=len(CLASSES))
    sd = torch.load(weights, map_location="cpu")
    net.load_state_dict({k.removeprefix("net."): v for k, v in sd.items()})
    return net.eval().to(device)


def preprocess(patches: np.ndarray) -> torch.Tensor:
    """(N, 200, 200) uint8 grayscale -> (N, 3, 224, 224) as in training (resize 224 + ImageNet norm)."""
    x = torch.from_numpy(patches).float().div(255)[:, None]
    x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
    return (x.expand(-1, 3, -1, -1) - _MEAN) / _STD


@torch.no_grad()
def landform_cell_probs(net: torch.nn.Module, img: np.ndarray, win: int = 200, stride: int = 50,
                        device: str = "cpu", batch: int = 64, max_nodata: float = 0.5) -> np.ndarray:
    """(15, n_r, n_c) class probabilities on a grid of stride x stride px cells covering
    [0, n*stride) of `img`: each cell averages every window overlapping it. Cells no window
    covers (nodata) are NaN. Windows with > max_nodata zero pixels are skipped."""
    assert win % stride == 0, "window must be a whole number of cells"
    k = win // stride
    rows, cols = range(0, img.shape[0] - win + 1, stride), range(0, img.shape[1] - win + 1, stride)
    acc = np.zeros((len(CLASSES), len(rows) + k - 1, len(cols) + k - 1), np.float32)
    cnt = np.zeros(acc.shape[1:], np.float32)
    coords = [(i, j) for i, r in enumerate(rows) for j, c in enumerate(cols)
              if (img[r:r + win, c:c + win] == 0).mean() <= max_nodata]
    for b in range(0, len(coords), batch):
        chunk = coords[b:b + batch]
        x = preprocess(np.stack([img[i * stride:i * stride + win, j * stride:j * stride + win] for i, j in chunk]))
        p = net(x.to(device)).softmax(1).cpu().numpy()
        for (i, j), pi in zip(chunk, p):
            acc[:, i:i + k, j:j + k] += pi[:, None, None]
            cnt[i:i + k, j:j + k] += 1
    with np.errstate(invalid="ignore"):
        return acc / cnt


def sample_to_grid(cells: np.ndarray, cell_transform: Affine, grid_transform: Affine,
                   grid_shape: tuple[int, int]) -> np.ndarray:
    """Value of the (north-up) cell grid at each target grid cell centre; UNKNOWN outside it."""
    rows, cols = np.arange(grid_shape[0]) + 0.5, np.arange(grid_shape[1]) + 0.5
    x = grid_transform.c + cols * grid_transform.a  # north-up grids: no rotation terms
    y = grid_transform.f + rows * grid_transform.e
    ci = np.floor((x - cell_transform.c) / cell_transform.a).astype(int)
    ri = np.floor((y - cell_transform.f) / cell_transform.e).astype(int)
    ok = (ri[:, None] >= 0) & (ri[:, None] < cells.shape[0]) & (ci[None] >= 0) & (ci[None] < cells.shape[1])
    out = np.full(grid_shape, UNKNOWN, cells.dtype)
    out[ok] = cells[np.clip(ri, 0, cells.shape[0] - 1)[:, None].repeat(len(ci), 1)[ok],
                    np.clip(ci, 0, cells.shape[1] - 1)[None].repeat(len(ri), 0)[ok]]
    return out
