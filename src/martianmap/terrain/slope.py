"""Slope from a DTM."""

import numpy as np
from scipy.ndimage import gaussian_filter, sobel


def filter_radius_px(sigma_px: float) -> int:
    """Pixels of context each slope value depends on (Gaussian truncate=4 plus the 3x3 Horn kernel)."""
    return (int(4 * sigma_px + 0.5) if sigma_px > 0 else 0) + 1


def slope_deg(dtm: np.ndarray, pixel_size_m: float, sigma_px: float) -> np.ndarray:
    """Slope in degrees: Gaussian pre-smooth, then Horn's method (= Sobel / 8)."""
    z = gaussian_filter(dtm.astype(np.float32), sigma_px) if sigma_px > 0 else dtm.astype(np.float32)
    gx = sobel(z, axis=1) / (8 * pixel_size_m)
    gy = sobel(z, axis=0) / (8 * pixel_size_m)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
