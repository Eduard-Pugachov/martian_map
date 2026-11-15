"""DTM-free mode: class-based cost and route comparison against the true-DTM route (PLAN §3E)."""

import numpy as np
from scipy.ndimage import binary_dilation

from martianmap.terrain.route import path_stats


def class_cost(cls: np.ndarray, class_costs: list[float]) -> np.ndarray:
    """Per-cell cost from hazard classes (0 safe, 1 caution, 2 hazard); any other value (255 = no data) -> inf."""
    lut = np.full(256, np.inf)
    lut[: len(class_costs)] = class_costs
    return lut[cls]


def overlap_pct(path: np.ndarray, ref: np.ndarray, shape: tuple[int, int], radius_cells: int) -> float:
    """% of `path` cells within `radius_cells` (8-connected) of a `ref` path cell."""
    near = np.zeros(shape, bool)
    near[ref[:, 0], ref[:, 1]] = True
    if radius_cells > 0:
        near = binary_dilation(near, structure=np.ones((3, 3), bool), iterations=radius_cells)
    return float(100 * near[path[:, 0], path[:, 1]].mean())


def compare(path: np.ndarray, ref: np.ndarray, true_cost: np.ndarray, slope: np.ndarray, dem: np.ndarray,
            cell_m: float, thresholds_deg: list[float], radius_cells: int) -> dict[str, float]:
    """Stats of `path` vs the reference (true-DTM) route, all measured on the true cost and true slope.

    extra_cost_pct is inf when the path enters a truly impassable cell; hazard_cells counts cells with
    true slope >= the hazard threshold.
    """
    st = path_stats(path, slope, dem, true_cost, cell_m, thresholds_deg[0])
    ref_st = path_stats(ref, slope, dem, true_cost, cell_m, thresholds_deg[0])
    s = slope[path[:, 0], path[:, 1]]
    return {
        "overlap_pct": overlap_pct(path, ref, true_cost.shape, radius_cells),
        "extra_cost_pct": 100 * (st["cost"] / ref_st["cost"] - 1),
        "hazard_cells": int(np.sum(s >= thresholds_deg[1])),
        "max_slope_deg": st["max_slope_deg"],
        "length_m": st["length_m"],
        "length_diff_m": st["length_m"] - ref_st["length_m"],
    }
