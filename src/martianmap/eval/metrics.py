"""Segmentation metrics from a confusion matrix."""

import torch


def confusion(pred: torch.Tensor, target: torch.Tensor, n_classes: int, ignore: int = 255) -> torch.Tensor:
    """(n, n) counts, rows = true class, cols = predicted; `ignore` targets are skipped."""
    keep = target != ignore
    idx = target[keep].long() * n_classes + pred[keep].long()
    return torch.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def iou_per_class(cm: torch.Tensor) -> torch.Tensor:
    """IoU per class; NaN for a class absent from both prediction and truth."""
    cm = cm.double()
    tp = cm.diag()
    union = cm.sum(0) + cm.sum(1) - tp
    return torch.where(union > 0, tp / union, torch.full_like(tp, float("nan")))
