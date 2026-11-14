"""Evaluate the hazard U-Net on a whole held-out region vs. two baselines, plus an aspect (illumination) table.

Usage: python scripts/eval_hazard.py [--split test|val]
Writes data/processed/eval/hazard_<split>.json and previews/eval_<split>.png.
"""

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter, sobel

from martianmap.config import load_config, repo_path
from martianmap.eval.baselines import fit_thresholds, gradient_magnitude
from martianmap.eval.metrics import confusion, iou_per_class
from martianmap.models.hazard_unet import load_model, predict_probs
from martianmap.terrain.hazard_labels import IGNORE

CLASSES = ("safe", "caution", "hazard")
PAL = np.array([[0.2, 0.6, 0.2], [0.95, 0.75, 0.1], [0.85, 0.1, 0.1]])
CONTEXT = 256  # extra image pixels read around the region so the model sees context at the region edge


def scores(pred: np.ndarray, labels: np.ndarray) -> dict:
    cm = confusion(torch.from_numpy(pred), torch.from_numpy(labels), 3, IGNORE)
    iou = iou_per_class(cm).tolist()
    return {"miou": float(np.nanmean(iou)), "iou": dict(zip(CLASSES, iou)),
            "pixel_acc": float(cm.diag().sum() / cm.sum()), "confusion": cm.tolist()}


def fit_gradient_baseline(cfg: dict, sigma: float, n_tiles: int = 150) -> tuple[float, float]:
    """Thresholds fit on a random sample of *train* tiles only."""
    p, tile = cfg["paths"], cfg["tiles"]["size_px"]
    with open(repo_path(p["tiles_index"]), newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
    rows = [rows[i] for i in np.random.default_rng(0).choice(len(rows), min(n_tiles, len(rows)), replace=False)]
    sc, lb = [], []
    with rasterio.open(repo_path(p["ortho_1m"])) as o, rasterio.open(repo_path(p["hazard_labels"])) as l:
        for r in rows:
            win = Window(int(r["col"]), int(r["row"]), tile, tile)
            img, lab = o.read(1, window=win), l.read(1, window=win)
            keep = (lab != IGNORE) & (img != 0)
            sc.append(gradient_magnitude(img / 255, sigma)[keep])
            lb.append(lab[keep])
    return fit_thresholds(np.concatenate(sc), np.concatenate(lb))


def downslope_quadrant(dtm: np.ndarray, sigma: float) -> np.ndarray:
    """0=N 1=E 2=S 3=W: compass direction each pixel's slope faces (downhill), from the smoothed DTM."""
    z = gaussian_filter(dtm, sigma)
    gx, gy = sobel(z, axis=1), sobel(z, axis=0)  # dz/d(east), dz/d(south)
    azimuth = np.degrees(np.arctan2(-gx, gy)) % 360  # downhill vector: east = -gx, north = +gy
    return (((azimuth + 45) % 360) // 90).astype(np.int8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--ckpt", default="checkpoints/hazard_unet_best.pt")
    args = ap.parse_args()
    cfg = load_config()
    p, t = cfg["paths"], cfg["tiles"]
    sigma = cfg["hazard"]["slope_sigma_px"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t1, t2 = fit_gradient_baseline(cfg, sigma)
    print(f"gradient baseline thresholds (fit on train tiles): {t1:.4f}, {t2:.4f}")

    r0, r1, c0, c1 = (v * t["block_m"] for v in t[f"{args.split}_blocks"])
    with rasterio.open(repo_path(p["ortho_1m"])) as o, rasterio.open(repo_path(p["hazard_labels"])) as l, \
         rasterio.open(repo_path(p["dtm_raw"])) as d:
        r1, c1 = min(r1, o.height), min(c1, o.width)
        cr0, cc0 = max(r0 - CONTEXT, 0), max(c0 - CONTEXT, 0)
        cr1, cc1 = min(r1 + CONTEXT, o.height), min(c1 + CONTEXT, o.width)
        img = o.read(1, window=Window(cc0, cr0, cc1 - cc0, cr1 - cr0))
        labels = l.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
        dtm = d.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
    inner = (slice(r0 - cr0, r0 - cr0 + labels.shape[0]), slice(c0 - cc0, c0 - cc0 + labels.shape[1]))
    labels[img[inner] == 0] = IGNORE
    print(f"{args.split} region rows {r0}-{r1}, cols {c0}-{c1} ({labels.size / 1e6:.0f} km^2), "
          f"{100 * (labels != IGNORE).mean():.0f}% labelled")

    probs = predict_probs(load_model(repo_path(args.ckpt), device), img / 255, device=device)[(slice(None), *inner)]
    preds = {
        "unet": probs.argmax(0).astype(np.uint8),
        "gradient": np.digitize(gradient_magnitude(img / 255, sigma)[inner], [t1, t2]).astype(np.uint8),
        "always_safe": np.zeros_like(labels),
    }
    results = {k: scores(v, labels) for k, v in preds.items()}

    print(f"\n{'method':12s} {'mIoU':>6s} {'safe':>6s} {'caution':>8s} {'hazard':>7s} {'pix acc':>8s}")
    for k, r in results.items():
        print(f"{k:12s} {r['miou']:6.3f} " + " ".join(f"{r['iou'][c]:{w}.3f}" for c, w in zip(CLASSES, (6, 8, 7)))
              + f" {r['pixel_acc']:8.3f}")
    cm = np.array(results["unet"]["confusion"])
    print("\nU-Net confusion (rows = DTM truth, cols = predicted, % of each true class):")
    for c, row in zip(CLASSES, cm):
        print(f"  {c:8s} " + " ".join(f"{100 * v / row.sum():6.1f}%" for v in row))

    # illumination check: same metrics split by the compass direction each slope faces
    quad = downslope_quadrant(dtm, sigma)
    aspect = {}
    print(f"\nBy slope-facing direction (DTM aspect):\n{'faces':6s} {'% px':>5s} {'unet mIoU':>10s} {'haz IoU':>8s} "
          f"{'haz recall':>11s} {'grad mIoU':>10s}")
    for q, name in enumerate("NESW"):
        m = quad == q
        lab_q = np.where(m, labels, IGNORE)
        u, g = scores(preds["unet"], lab_q), scores(preds["gradient"], lab_q)
        haz = np.array(u["confusion"])[2]
        aspect[name] = {"pct_px": float(100 * m[labels != IGNORE].mean()), "unet": u, "gradient": g,
                        "hazard_recall": float(haz[2] / max(haz.sum(), 1))}
        print(f"{name:6s} {aspect[name]['pct_px']:5.1f} {u['miou']:10.3f} {u['iou']['hazard']:8.3f} "
              f"{aspect[name]['hazard_recall']:11.3f} {g['miou']:10.3f}")

    out_dir = repo_path(p["processed"]) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"hazard_{args.split}.json"
    out.write_text(json.dumps({"split": args.split, "ckpt": args.ckpt, "gradient_thresholds": [t1, t2],
                               "region_px": [r0, r1, c0, c1], "results": results, "by_aspect": aspect}, indent=2))
    print(f"\nwrote {out}")

    k = 4  # thumbnail factor
    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    ax[0].imshow(img[inner][::k, ::k], cmap="gray")
    show = lambda a: np.where((labels == IGNORE)[::k, ::k, None], 0.3, PAL[np.minimum(a, 2)[::k, ::k]])
    ax[1].imshow(show(labels))
    ax[2].imshow(show(preds["unet"]))
    ax[3].imshow(show(preds["gradient"]))
    for a, title in zip(ax, ("ortho (1 m)", "DTM truth", f"U-Net (mIoU {results['unet']['miou']:.2f})",
                             f"gradient baseline (mIoU {results['gradient']['miou']:.2f})")):
        a.set_title(title)
        a.axis("off")
    fig.tight_layout()
    png = repo_path(p["previews"]) / f"eval_{args.split}.png"
    fig.savefig(png, dpi=80)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
