"""Least-cost routes and path statistics."""

import numpy as np
from skimage.draw import line
from skimage.graph import route_through_array


def least_cost_route(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    """(N, 2) row/col path minimising summed cost (8-connected, diagonal steps cost sqrt(2)).

    Raises ValueError if the goal is unreachable (inf = impassable).
    """
    for name, (r, c) in (("start", start), ("goal", goal)):
        if not np.isfinite(cost[r, c]):
            raise ValueError(f"{name} {(r, c)} is on an impassable or no-data cell")
    path, _ = route_through_array(cost, start, goal, fully_connected=True, geometric=True)
    return np.asarray(path)


def straight_line(start: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    """(N, 2) row/col cells on the straight line from start to goal."""
    return np.stack(line(*start, *goal), axis=1)


def path_stats(path: np.ndarray, slope: np.ndarray, dem: np.ndarray, cost: np.ndarray, cell_m: float,
               caution_deg: float) -> dict[str, float]:
    """Length, climb, slope and hazard figures along a path; summed cost uses the true cost map."""
    s = slope[path[:, 0], path[:, 1]]
    z = dem[path[:, 0], path[:, 1]]
    c = cost[path[:, 0], path[:, 1]]
    steps = np.hypot(*np.diff(path, axis=0).T) * cell_m
    blocked = ~np.isfinite(c)
    return {
        "length_m": float(steps.sum()),
        "climb_m": float(np.nansum(np.clip(np.diff(z), 0, None))),
        "max_slope_deg": float(np.nanmax(s)) if np.isfinite(s).any() else float("nan"),
        "pct_caution_or_worse": float(100 * np.mean(blocked | (s >= caution_deg))),
        "impassable_cells": int(blocked.sum()),
        # step cost = mean of the two cells' costs x step length (how route_through_array charges it)
        "cost": float(np.sum((c[:-1] + c[1:]) / 2 * steps)),
    }
