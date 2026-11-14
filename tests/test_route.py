import numpy as np
import pytest

from martianmap.terrain.route import least_cost_route, path_stats, straight_line


def wall_with_gap() -> np.ndarray:
    cost = np.ones((7, 7))
    cost[:, 3] = np.inf
    cost[6, 3] = 1.0  # only way through is the bottom row
    return cost


def test_route_goes_through_gap_and_avoids_walls() -> None:
    cost = wall_with_gap()
    path = least_cost_route(cost, (0, 0), (0, 6))
    assert tuple(path[0]) == (0, 0) and tuple(path[-1]) == (0, 6)
    assert np.isfinite(cost[path[:, 0], path[:, 1]]).all()
    assert (6, 3) in map(tuple, path)


def test_route_prefers_cheap_detour() -> None:
    cost = np.ones((5, 9))
    cost[0:3, 4] = 50.0  # expensive band, cheap gap at the bottom
    path = least_cost_route(cost, (0, 0), (0, 8))
    assert cost[path[:, 0], path[:, 1]].max() == 1.0


def test_unreachable_or_blocked_endpoints_raise() -> None:
    cost = np.ones((3, 3))
    cost[:, 1] = np.inf
    with pytest.raises(ValueError):
        least_cost_route(cost, (0, 0), (0, 2))
    with pytest.raises(ValueError, match="goal"):
        least_cost_route(np.ones((3, 3)) + np.diag([0, 0, np.inf]), (0, 0), (2, 2))


def test_straight_line_endpoints() -> None:
    p = straight_line((0, 0), (3, 6))
    assert tuple(p[0]) == (0, 0) and tuple(p[-1]) == (3, 6) and len(p) == 7


def test_path_stats() -> None:
    path = np.array([[0, 0], [1, 1], [1, 2]])
    slope = np.array([[5.0, 0, 0], [0, 12.0, 30.0]])
    dem = np.array([[100.0, 0, 0], [0, 103.0, 101.0]])
    cost = np.array([[1.0, 1, 1], [1, 2.0, np.inf]])
    s = path_stats(path, slope, dem, cost, cell_m=5.0, caution_deg=10.0)
    assert s["length_m"] == pytest.approx(5 * (np.sqrt(2) + 1))
    assert s["climb_m"] == 3.0 and s["max_slope_deg"] == 30.0
    assert s["impassable_cells"] == 1 and s["pct_caution_or_worse"] == pytest.approx(200 / 3)
