"""Precompute map layers from the DTM -> data/processed.

Usage: python scripts/precompute_layers.py --layers slope hazard --preview
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from martianmap.config import load_config, repo_path
from martianmap.terrain.hazard_labels import IGNORE, hazard_classes, invalid_mask
from martianmap.terrain.slope import filter_radius_px, slope_deg

STRIP_ROWS = 2048  # ~175 MB per float32 strip; the full DTM at once needs ~9 GB peak


def slope_and_hazard(cfg: dict) -> None:
    """Slope (deg) and hazard labels for the whole DTM, processed in overlapping row strips."""
    p, hz = cfg["paths"], cfg["hazard"]
    sigma, thresholds = hz["slope_sigma_px"], hz["thresholds_deg"]
    halo = filter_radius_px(sigma)
    out_slope, out_labels = repo_path(p["slope"]), repo_path(p["hazard_labels"])
    out_slope.parent.mkdir(parents=True, exist_ok=True)

    class_counts = np.zeros(256, np.int64)
    slope_hist = np.zeros(90, np.int64)  # 1-degree bins over valid pixels
    with rasterio.open(repo_path(p["dtm_raw"])) as src:
        px = src.res[0]
        assert abs(src.res[0] - src.res[1]) < 1e-9, "non-square DTM pixels"
        common = dict(driver="GTiff", width=src.width, height=src.height, count=1, crs=src.crs,
                      transform=src.transform, tiled=True, blockxsize=512, blockysize=512, compress="deflate", BIGTIFF="IF_SAFER")
        with rasterio.open(out_slope, "w", dtype="float32", nodata=np.nan, predictor=3, **common) as dst_s, \
             rasterio.open(out_labels, "w", dtype="uint8", nodata=IGNORE, **common) as dst_l:
            for r0 in tqdm(range(0, src.height, STRIP_ROWS), desc="slope+hazard strips"):
                r1 = min(r0 + STRIP_ROWS, src.height)
                a0, a1 = max(r0 - halo, 0), min(r1 + halo, src.height)
                dtm = src.read(1, window=Window(0, a0, src.width, a1 - a0))
                invalid = invalid_mask(dtm, src.nodata)
                dtm[invalid] = 0.0  # filler; everything it touches is masked by the halo-wide margin
                slope = slope_deg(dtm, px, sigma)
                labels = hazard_classes(slope, invalid, thresholds, margin_px=halo)
                core = slice(r0 - a0, r0 - a0 + (r1 - r0))
                slope, labels = slope[core], labels[core]
                slope[labels == IGNORE] = np.nan
                dst_s.write(slope, 1, window=Window(0, r0, src.width, r1 - r0))
                dst_l.write(labels, 1, window=Window(0, r0, src.width, r1 - r0))
                class_counts += np.bincount(labels.ravel(), minlength=256)
                slope_hist += np.histogram(slope[labels != IGNORE], bins=90, range=(0, 90))[0]

    valid = class_counts[:3].sum()
    total = class_counts.sum()
    print(f"\nDTM {total:,} px, valid {valid:,} ({100 * valid / total:.1f}%), ignore {class_counts[IGNORE]:,}")
    for i, name in enumerate(["safe", "caution", "hazard"]):
        print(f"  {i} {name:8s} {class_counts[i]:>13,}  {100 * class_counts[i] / valid:6.2f}% of valid")
    cum = slope_hist[::-1].cumsum()[::-1]  # cum[d] = valid px with slope >= d
    print("  slope >= " + "  ".join(f"{d}deg: {100 * cum[d] / valid:.2f}%" for d in (15, 20, 25, 30, 40)))
    print(f"wrote {out_slope}\nwrote {out_labels}")


def preview(cfg: dict, scale: int = 16) -> None:
    """Downsampled PNG of slope and hazard labels (nearest-neighbour, so thin scarps may vanish)."""
    p = cfg["paths"]
    out = repo_path(p["previews"]) / "slope_hazard.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(repo_path(p["slope"])) as s, rasterio.open(repo_path(p["hazard_labels"])) as l:
        shape = (s.height // scale, s.width // scale)
        slope, labels = s.read(1, out_shape=shape), l.read(1, out_shape=shape)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    im = ax[0].imshow(slope, cmap="magma", vmin=0, vmax=35)
    fig.colorbar(im, ax=ax[0], shrink=0.7, label="slope (deg)")
    rgb = np.zeros((*labels.shape, 3))
    for k, c in {0: (0.2, 0.6, 0.2), 1: (0.95, 0.75, 0.1), 2: (0.85, 0.1, 0.1), IGNORE: (0.3, 0.3, 0.3)}.items():
        rgb[labels == k] = c
    ax[1].imshow(rgb)
    ax[0].set_title("Slope (deg)")
    ax[1].set_title("Hazard: green safe / yellow caution / red hazard / grey ignore")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", nargs="+", choices=["slope", "hazard"], required=True)
    ap.add_argument("--preview", action="store_true", help="write a downsampled PNG to data/processed/previews")
    args = ap.parse_args()
    cfg = load_config()
    # slope and hazard come out of the same pass over the DTM
    slope_and_hazard(cfg)
    if args.preview:
        preview(cfg)


if __name__ == "__main__":
    main()
