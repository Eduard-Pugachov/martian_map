import numpy as np

from martianmap.terrain.compare_routes import class_cost, compare, overlap_pct


def test_class_cost_lookup_and_nodata() -> None:
    cls = np.array([[0, 1, 2, 255]], np.uint8)
    c = class_cost(cls, [1.0, 3.0, 50.0])
    assert c[0, :3].tolist() == [1.0, 3.0, 50.0] and np.isinf(c[0, 3])


def test_overlap_radius() -> None:
    ref = np.array([[0, 0], [0, 1], [0, 2]])
    path = np.array([[0, 0], [1, 1], [2, 2]])  # 1 on ref, 1 one cell away, 1 two cells away
    assert np.isclose(overlap_pct(path, ref, (3, 3), 0), 100 / 3)
    assert np.isclose(overlap_pct(path, ref, (3, 3), 1), 200 / 3)


def test_compare_identical_and_detour() -> None:
    slope = np.zeros((3, 4))
    slope[0, 1:3] = 30.0  # true hazard in the top row
    cost = np.where(slope >= 25, np.inf, 1.0)
    dem = np.zeros_like(slope)
    ref = np.array([[1, 0], [1, 1], [1, 2], [1, 3]])
    same = compare(ref, ref, cost, slope, dem, 5.0, [10.0, 25.0], 1)
    assert same["overlap_pct"] == 100 and same["extra_cost_pct"] == 0 and same["hazard_cells"] == 0
    bad = compare(np.array([[1, 0], [0, 1], [0, 2], [1, 3]]), ref, cost, slope, dem, 5.0, [10.0, 25.0], 0)
    assert bad["hazard_cells"] == 2 and np.isinf(bad["extra_cost_pct"]) and bad["max_slope_deg"] == 30.0
    assert bad["length_diff_m"] > 0
