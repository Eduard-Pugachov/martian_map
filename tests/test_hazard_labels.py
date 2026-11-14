import numpy as np

from martianmap.terrain.hazard_labels import CAUTION, HAZARD, IGNORE, SAFE, hazard_classes, invalid_mask
from martianmap.terrain.slope import filter_radius_px, slope_deg


def ramp(deg: float, n: int = 64) -> np.ndarray:
    """Plane rising along x at `deg` degrees, 1 m pixels."""
    return np.tile(np.arange(n, dtype=np.float32) * np.tan(np.radians(deg)), (n, 1))


def test_slope_of_plane_is_exact_in_interior() -> None:
    for deg in (0.0, 5.0, 30.0, 60.0):
        s = slope_deg(ramp(deg), pixel_size_m=1.0, sigma_px=1.0)
        assert np.allclose(s[10:-10, 10:-10], deg, atol=1e-3)


def test_pixel_size_scales_slope() -> None:
    s = slope_deg(ramp(45.0), pixel_size_m=2.0, sigma_px=0.0)  # same rise over 2 m pixels -> tan = 0.5
    assert np.allclose(s[5:-5, 5:-5], np.degrees(np.arctan(0.5)), atol=1e-3)


def test_threshold_boundaries() -> None:
    slope = np.array([[0.0, 9.99, 10.0, 24.99, 25.0, 80.0]], np.float32)
    labels = hazard_classes(slope, np.zeros_like(slope, bool), [10.0, 25.0], margin_px=0)
    assert labels.tolist() == [[SAFE, SAFE, CAUTION, CAUTION, HAZARD, HAZARD]]


def test_nodata_and_margin_are_ignored() -> None:
    nodata = -3.4028226550889045e38
    dtm = np.zeros((20, 20), np.float32)
    dtm[10, 10] = nodata
    dtm[0, 0] = np.nan
    invalid = invalid_mask(dtm, nodata)
    labels = hazard_classes(np.zeros_like(dtm), invalid, [10.0, 25.0], margin_px=2)
    assert labels[10, 10] == IGNORE and labels[8, 12] == IGNORE  # within 2 px (chessboard)
    assert labels[7, 10] == SAFE and labels[0, 0] == IGNORE
    assert set(np.unique(labels)) <= {SAFE, CAUTION, HAZARD, IGNORE}


def test_filter_radius() -> None:
    assert filter_radius_px(1.0) == 5 and filter_radius_px(0.0) == 1
