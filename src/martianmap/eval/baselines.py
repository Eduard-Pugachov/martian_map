"""Non-learned baselines for image -> hazard class."""

import numpy as np
from scipy.ndimage import gaussian_filter, sobel


def gradient_magnitude(img01: np.ndarray, sigma_px: float = 1.0) -> np.ndarray:
    """Brightness-gradient magnitude after the same Gaussian smoothing the slope labels use."""
    z = gaussian_filter(img01.astype(np.float32), sigma_px)
    return np.hypot(sobel(z, axis=0), sobel(z, axis=1))


def fit_thresholds(score: np.ndarray, labels: np.ndarray, n_bins: int = 200) -> tuple[float, float]:
    """Two score thresholds (safe < t1 <= caution < t2 <= hazard) maximising mIoU on labelled pixels.

    Exhaustive search over score-quantile bin edges, using cumulative per-class histograms.
    """
    edges = np.unique(np.quantile(score, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, len(edges) - 2)
    hist = np.stack([np.bincount(idx[labels == k], minlength=len(edges) - 1) for k in range(3)])  # (3, bins)
    cum = np.concatenate([np.zeros((3, 1)), hist.cumsum(1)], axis=1)  # cum[:, i] = count in bins < i
    n = cum.shape[1]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    # cm[true, pred] for every (i, j): pred 0 below edge i, 1 between, 2 from edge j up
    cm = np.stack([cum[:, i], cum[:, j] - cum[:, i], cum[:, -1:, None] - cum[:, j]], axis=1)  # (3, 3, n, n)
    tp = cm[[0, 1, 2], [0, 1, 2]]
    union = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        miou = np.nanmean(np.where(union > 0, tp / union, np.nan), axis=0)
    miou[j < i] = -1
    bi, bj = np.unravel_index(np.nanargmax(miou), miou.shape)
    return float(edges[min(bi, len(edges) - 1)]), float(edges[min(bj, len(edges) - 1)])
