"""Slope -> safe / caution / hazard classes, with an ignore mask."""

import numpy as np
from scipy.ndimage import binary_dilation

SAFE, CAUTION, HAZARD, IGNORE = 0, 1, 2, 255


def invalid_mask(dtm: np.ndarray, nodata: float | None) -> np.ndarray:
    """True where the DTM has no data."""
    bad = ~np.isfinite(dtm)
    if nodata is not None:
        bad |= dtm == np.float32(nodata)
    return bad


def hazard_classes(slope: np.ndarray, invalid: np.ndarray, thresholds_deg: list[float], margin_px: int) -> np.ndarray:
    """Classes 0/1/2 by slope thresholds; IGNORE on nodata and within margin_px of it (filter contamination)."""
    labels = np.digitize(slope, thresholds_deg).astype(np.uint8)
    if margin_px > 0 and invalid.any():
        invalid = binary_dilation(invalid, structure=np.ones((3, 3), bool), iterations=margin_px)
    labels[invalid] = IGNORE
    return labels
