import numpy as np

from martianmap.terrain.compare_routes import class_cost, compare, overlap_pct, risk_classes, wilson_ci95


def test_risk_classes_threshold_rule() -> None:
    cls = np.array([[0, 1, 2, 0, 255, 1]], np.uint8)
    p = np.array([[0.05, 0.2, 0.9, 0.3, 0.9, np.nan]], np.float16)
    out = risk_classes(cls, p, 0.2)
    assert out.tolist() == [[0, 2, 2, 2, 255, 1]]  # >= t -> hazard, < t unchanged, no-data kept, NaN ignored
    assert cls.tolist() == [[0, 1, 2, 0, 255, 1]]  # input not modified
    assert class_cost(out, [1.0, 3.0, 50.0])[0, 1] == 50.0
    assert (risk_classes(cls, p, 0.1) == 2).sum() >= (out == 2).sum()  # lower t never removes hazard


def test_wilson_ci95() -> None:
    lo, hi = wilson_ci95(4, 20)  # textbook value: 8.1% - 41.6%
    assert np.isclose(lo, 8.07, atol=0.05) and np.isclose(hi, 41.6, atol=0.05)
    assert wilson_ci95(0, 50)[0] == 0 and wilson_ci95(50, 50)[1] == 100


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
