import numpy as np

from martianmap.terrain.costmap import block_max, block_mean, slope_cost


def test_slope_cost_values() -> None:
    cost = slope_cost(np.array([0.0, 10.0, 20.0, 24.9, 25.0, 60.0, np.nan]), scale_deg=10, max_slope_deg=25)
    assert np.allclose(cost[:4], [1.0, 2.0, 5.0, 1 + 2.49**2])
    assert np.isinf(cost[4:]).all()


def test_block_max_keeps_spikes_and_nan() -> None:
    a = np.zeros((10, 11), np.float32)
    a[1, 1] = 30.0
    a[7, 7] = np.nan
    m = block_max(a, 5)
    assert m.shape == (2, 2)  # ragged column trimmed
    assert m[0, 0] == 30.0 and np.isnan(m[1, 1]) and m[0, 1] == 0.0


def test_block_mean() -> None:
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert block_mean(a, 2).tolist() == [[2.5, 4.5], [10.5, 12.5]]
