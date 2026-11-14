"""Precompute map layers from the DTM -> data/processed.

Usage:
  python scripts/precompute_layers.py --layers slope hazard --preview   # M1: 1 m slope + hazard labels
  python scripts/precompute_layers.py --layers cost --route --preview   # M2: 5 m routing grid + demo route
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from matplotlib.colors import LightSource
from rasterio.transform import Affine, rowcol

from martianmap.config import load_config, repo_path
from martianmap.terrain.costmap import block_max, block_mean, slope_cost
from martianmap.terrain.hazard_labels import IGNORE, hazard_classes, invalid_mask
from martianmap.terrain.route import least_cost_route, path_stats, straight_line
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


def route_grid(cfg: dict) -> None:
    """Downsample 1 m slope (block max) and DTM (block mean) to the routing cell size -> .npz."""
    p = cfg["paths"]
    f = round(cfg["grid"]["route_cell_size_m"] / cfg["grid"]["dtm_pixel_size_m"])
    rows = (STRIP_ROWS // f) * f  # strips must hold whole blocks
    slopes, dems = [], []
    with rasterio.open(repo_path(p["slope"])) as s, rasterio.open(repo_path(p["dtm_raw"])) as d:
        assert s.transform == d.transform and s.shape == d.shape, "slope and DTM grids differ"
        height = (s.height // f) * f
        for r0 in tqdm(range(0, height, rows), desc=f"{f}x{f} routing grid"):
            win = Window(0, r0, s.width, min(rows, height - r0))
            slopes.append(block_max(s.read(1, window=win), f))
            dem = d.read(1, window=win)
            dem[invalid_mask(dem, d.nodata)] = np.nan
            dems.append(block_mean(dem, f))
        transform = s.transform * Affine.scale(f)
    out = repo_path(p["route_grid"])
    np.savez_compressed(out, slope=np.vstack(slopes).astype(np.float32), dem=np.vstack(dems).astype(np.float32),
                        transform=np.array(transform)[:6], cell_m=f * s.res[0])
    print(f"wrote {out}  grid {np.vstack(slopes).shape}, cell {f * s.res[0]:g} m")


def demo_route(cfg: dict, preview_png: bool) -> None:
    """Least-cost vs straight-line route between the config demo endpoints, with stats (+ optional PNG)."""
    g = np.load(repo_path(cfg["paths"]["route_grid"]))
    slope, dem, cell = g["slope"], g["dem"], float(g["cell_m"])
    transform = Affine(*g["transform"])
    c = cfg["cost"]
    cost = slope_cost(slope, c["slope_scale_deg"], c["max_slope_deg"], round(c["hazard_buffer_m"] / cell))
    start, goal = (tuple(int(v) for v in rowcol(transform, *cfg["route"][k])) for k in ("start_xy", "goal_xy"))
    for name, (r, col) in (("start", start), ("goal", goal)):
        if not (0 <= r < cost.shape[0] and 0 <= col < cost.shape[1]):
            raise SystemExit(f"{name} {cfg['route'][name + '_xy']} is outside the routing grid")
    paths = {"safest": least_cost_route(cost, start, goal), "straight": straight_line(start, goal)}
    caution = cfg["hazard"]["thresholds_deg"][0]
    stats = {k: path_stats(v, slope, dem, cost, cell, caution) for k, v in paths.items()}
    print(f"\n{'':10s}" + "".join(f"{k:>24s}" for k in stats["safest"]))
    for k, st in stats.items():
        print(f"{k:10s}" + "".join(f"{v:24.1f}" for v in st.values()))
    if not preview_png:
        return
    out = repo_path(cfg["paths"]["previews"]) / "route.png"
    shade = LightSource(azdeg=315, altdeg=45).hillshade(np.nan_to_num(dem, nan=np.nanmean(dem)), dx=cell, dy=cell)
    shade[np.isnan(dem)] = np.nan
    blocked = np.ma.masked_where(np.isfinite(cost), np.ones_like(cost))
    (r0, c0), (r1, c1) = np.vstack(list(paths.values())).min(0) - 60, np.vstack(list(paths.values())).max(0) + 60
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    for a, zoom in zip(ax, (False, True)):
        a.imshow(shade, cmap="gray")
        a.imshow(blocked, cmap="autumn", alpha=0.6, interpolation="nearest")
        for (k, pth), style in zip(paths.items(), ("r-", "c--")):
            a.plot(pth[:, 1], pth[:, 0], style, lw=1.5, label=f"{k}: {stats[k]['length_m']:.0f} m")
        a.plot(*start[::-1], "go", ms=8)
        a.plot(*goal[::-1], "b*", ms=12)
        if zoom:
            a.set_xlim(max(c0, 0), c1)
            a.set_ylim(r1, max(r0, 0))
        a.axis("off")
    ax[0].set_title(f"{cell:g} m grid; red/orange = impassable (slope >= {c['max_slope_deg']:g} deg or no data)")
    ax[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", nargs="*", default=[], choices=["slope", "hazard", "cost"])
    ap.add_argument("--route", action="store_true", help="route between config route.start_xy and goal_xy")
    ap.add_argument("--preview", action="store_true", help="write downsampled PNGs to data/processed/previews")
    args = ap.parse_args()
    cfg = load_config()
    if {"slope", "hazard"} & set(args.layers):  # both come out of the same pass over the DTM
        slope_and_hazard(cfg)
        if args.preview:
            preview(cfg)
    if "cost" in args.layers:  # cost itself is cheap and computed at route time from the grid's slope
        route_grid(cfg)
    if args.route:
        demo_route(cfg, args.preview)


if __name__ == "__main__":
    main()
