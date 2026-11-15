"""DTM-free mode: class-based cost and route comparison against the true-DTM route (PLAN §3E)."""

import numpy as np
from scipy.ndimage import binary_dilation

from martianmap.terrain.costmap import buffer_impassable, slope_cost
from martianmap.terrain.route import path_stats


def class_cost(cls: np.ndarray, class_costs: list[float]) -> np.ndarray:
    """Per-cell cost from hazard classes (0 safe, 1 caution, 2 hazard); any other value (255 = no data) -> inf."""
    lut = np.full(256, np.inf)
    lut[: len(class_costs)] = class_costs
    return lut[cls]


def build_costs(slope: np.ndarray, pred: np.ndarray, p_hazard: np.ndarray, mult: np.ndarray, cfg: dict,
                cell_m: float, thresholds: list[float]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Cost maps T (true slope), O (oracle classes), P (U-Net argmax), L (landform only) and P@t per threshold,
    all x landform multiplier with the impassable buffer. Also returns the unbuffered true cost for scoring
    (the buffer is a planning margin, not ground truth)."""
    c, d = cfg["cost"], cfg["dtm_free"]
    buf = round(c["hazard_buffer_m"] / cell_m)
    with np.errstate(invalid="ignore"):
        oracle = np.where(np.isfinite(slope), np.digitize(slope, cfg["hazard"]["thresholds_deg"]), 255).astype(np.uint8)
    true = slope_cost(slope, c["slope_scale_deg"], c["max_slope_deg"]) * mult
    costs = {
        "T": buffer_impassable(true, buf),
        "O": buffer_impassable(class_cost(oracle, d["class_costs"]) * mult, buf),
        "P": buffer_impassable(class_cost(pred, d["class_costs"]) * mult, buf),
        "L": buffer_impassable(np.where(pred == 255, np.inf, 1.0) * mult, buf),
    }
    for t in thresholds:
        costs[f"P@{t:g}"] = buffer_impassable(class_cost(risk_classes(pred, p_hazard, t), d["class_costs"]) * mult, buf)
    return costs, true


def risk_classes(cls: np.ndarray, p_hazard: np.ndarray, t: float) -> np.ndarray:
    """Risk-aware classes: hazard (2) wherever p_hazard >= t, else the argmax class. No-data (255) stays
    no-data; NaN probabilities never trigger."""
    out = cls.copy()
    with np.errstate(invalid="ignore"):
        out[(p_hazard >= t) & (cls != 255)] = 2
    return out


def overlap_pct(path: np.ndarray, ref: np.ndarray, shape: tuple[int, int], radius_cells: int) -> float:
    """% of `path` cells within `radius_cells` (8-connected) of a `ref` path cell."""
    near = np.zeros(shape, bool)
    near[ref[:, 0], ref[:, 1]] = True
    if radius_cells > 0:
        near = binary_dilation(near, structure=np.ones((3, 3), bool), iterations=radius_cells)
    return float(100 * near[path[:, 0], path[:, 1]].mean())


def wilson_ci95(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval (in %) for k successes out of n; sane at k = 0 or k = n."""
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * max(centre - half, 0.0), 100 * min(centre + half, 1.0)


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
