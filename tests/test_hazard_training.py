import numpy as np
import torch

from martianmap.data.hazard_dataset import augment, to_input
from martianmap.eval.metrics import confusion, iou_per_class


def test_confusion_skips_ignore_and_iou() -> None:
    target = torch.tensor([0, 0, 1, 2, 255])
    pred = torch.tensor([0, 1, 1, 2, 0])
    cm = confusion(pred, target, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert torch.allclose(iou_per_class(cm), torch.tensor([0.5, 0.5, 1.0], dtype=torch.float64))


def test_iou_nan_for_absent_class() -> None:
    cm = confusion(torch.tensor([0, 0]), torch.tensor([0, 0]), 3)
    iou = iou_per_class(cm)
    assert iou[0] == 1.0 and torch.isnan(iou[1:]).all()


def test_augment_keeps_image_and_label_aligned() -> None:
    lab = np.arange(64, dtype=np.uint8).reshape(8, 8)
    img = 0.4 + lab.astype(np.float32) / 640  # 0.40-0.50: never clipped, so values stay distinct
    for seed in range(20):
        torch.manual_seed(seed)
        a_img, a_lab = augment(img.copy(), lab.copy())
        # photometric change is monotonic, so pixel ordering of image must match label ordering
        assert (np.argsort(a_img.ravel(), kind="stable") == np.argsort(a_lab.ravel(), kind="stable")).all()
        assert 0 <= a_img.min() and a_img.max() <= 1


def test_to_input_shape_and_channels() -> None:
    x = to_input(np.full((4, 5), 0.5, np.float32))
    assert x.shape == (3, 4, 5) and not torch.equal(x[0], x[1])  # per-channel ImageNet norm
