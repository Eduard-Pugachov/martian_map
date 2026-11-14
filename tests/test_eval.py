import numpy as np
import torch

from martianmap.eval.baselines import fit_thresholds, gradient_magnitude
from martianmap.models.hazard_unet import predict_probs, window_starts


def test_fit_thresholds_recovers_a_perfect_split() -> None:
    rng = np.random.default_rng(0)
    score = rng.uniform(0, 1, 50_000)
    labels = np.digitize(score, [0.6, 0.9])
    t1, t2 = fit_thresholds(score, labels)
    assert abs(t1 - 0.6) < 0.01 and abs(t2 - 0.9) < 0.01
    assert (np.digitize(score, [t1, t2]) == labels).mean() > 0.99


def test_gradient_magnitude_flat_is_zero_and_edge_is_positive() -> None:
    img = np.zeros((20, 20), np.float32)
    assert gradient_magnitude(img).max() == 0
    img[:, 10:] = 1
    g = gradient_magnitude(img)
    assert g[10, 10] > 0 and g[10, 2] < 1e-3


def test_window_starts_cover_and_end_flush() -> None:
    assert window_starts(1000, 512, 256) == [0, 256, 488]
    assert window_starts(512, 512, 256) == [0]


class Const(torch.nn.Module):
    """Fake model: logits favour class 2 everywhere."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], 3, *x.shape[2:])
        out[:, 2] = 5.0
        return out


def test_predict_probs_shape_and_normalisation() -> None:
    p = predict_probs(Const(), np.full((70, 90), 0.5, np.float32), tile=32, stride=16, device="cpu")
    assert p.shape == (3, 70, 90)
    assert np.allclose(p.sum(0), 1, atol=1e-5) and (p.argmax(0) == 2).all()
