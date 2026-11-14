"""Routing grid and cost map."""

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import block_reduce


def block_max(a: np.ndarray, factor: int) -> np.ndarray:
    """Max over factor x factor blocks (any NaN in a block -> NaN). Trims ragged edges.

    Max, not mean, so a 1 m scarp can't be averaged away. Conservative: can close narrow passages.
    """
    h, w = (a.shape[0] // factor) * factor, (a.shape[1] // factor) * factor
    return block_reduce(a[:h, :w], (factor, factor), np.max)


def block_mean(a: np.ndarray, factor: int) -> np.ndarray:
    """Mean over factor x factor blocks (any NaN in a block -> NaN). Trims ragged edges."""
    h, w = (a.shape[0] // factor) * factor, (a.shape[1] // factor) * factor
    return block_reduce(a[:h, :w], (factor, factor), np.mean)


def slope_cost(slope_deg: np.ndarray, scale_deg: float, max_slope_deg: float, buffer_cells: int = 0) -> np.ndarray:
    """Per-cell cost 1 + (slope/scale)^2; inf (impassable) at >= max_slope_deg or no data,
    plus a safety buffer of `buffer_cells` around those cells (8-connected)."""
    with np.errstate(invalid="ignore"):
        cost = 1.0 + (slope_deg / scale_deg) ** 2
        blocked = ~np.isfinite(slope_deg) | (slope_deg >= max_slope_deg)
    if buffer_cells > 0:
        blocked = binary_dilation(blocked, structure=np.ones((3, 3), bool), iterations=buffer_cells)
    cost[blocked] = np.inf
    return cost
