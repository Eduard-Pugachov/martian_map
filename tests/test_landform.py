import numpy as np
import torch
from rasterio.transform import Affine

from martianmap.models.landform import CLASSES, UNKNOWN, landform_cell_probs, sample_to_grid
from martianmap.terrain.costmap import buffer_impassable, landform_multiplier


class BrightnessNet(torch.nn.Module):
    """Fake classifier: class 0 for dark patches, class 13 ('smo') for bright ones."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], len(CLASSES))
        bright = x.mean((1, 2, 3)) > 0
        out[bright, 13] = 10.0
        out[~bright, 0] = 10.0
        return out


def test_cell_probs_grid_shape_normalised_and_spatially_right() -> None:
    img = np.full((400, 600), 20, np.uint8)
    img[:, 400:] = 240  # bright right third
    p = landform_cell_probs(BrightnessNet(), img, win=200, stride=50)
    assert p.shape == (15, 8, 12)  # (400-200)/50+1 + 3 rows, (600-200)/50+1 + 3 cols
    assert np.allclose(p.sum(0), 1, atol=1e-5)
    cls = p.argmax(0)
    assert (cls[:, :4] == 0).all() and (cls[:, -2:] == 13).all()


def test_cell_probs_skip_nodata_windows() -> None:
    img = np.zeros((200, 400), np.uint8)
    img[:, :200] = 100
    p = landform_cell_probs(BrightnessNet(), img, win=200, stride=50)
    assert np.isfinite(p[:, :, 0]).all() and np.isnan(p[:, :, -1]).all()


def test_sample_to_grid_maps_coordinates() -> None:
    cells = np.array([[1, 2], [3, 4]], np.uint8)  # 300 m cells, origin (1000, 2000)
    cell_t = Affine(300, 0, 1000, 0, -300, 2000)
    grid_t = Affine(150, 0, 1000, 0, -150, 2000)  # 150 m grid: 2x2 target cells per source cell
    out = sample_to_grid(cells, cell_t, grid_t, (5, 5))
    assert out[:4, :4].tolist() == [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
    assert (out[4] == UNKNOWN).all() and (out[:, 4] == UNKNOWN).all()


def test_landform_multiplier_and_buffer() -> None:
    cls = np.array([[CLASSES.index("smo"), CLASSES.index("cli"), CLASSES.index("aec"), UNKNOWN]], np.uint8)
    m = landform_multiplier(cls, CLASSES, {"smo": 1.0, "cli": None, "aec": 4.0})
    assert m[0, 0] == 1.0 and np.isinf(m[0, 1]) and m[0, 2] == 4.0 and m[0, 3] == 1.0
    cost = buffer_impassable(np.ones((1, 4)) * m, 1)
    assert np.isinf(cost[0, :3]).all() and cost[0, 3] == 1.0
