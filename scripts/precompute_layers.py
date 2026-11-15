"""Precompute map layers from the DTM -> data/processed.

Usage:
  python scripts/precompute_layers.py --layers slope hazard --preview   # M1: 1 m slope + hazard labels
  python scripts/precompute_layers.py --layers cost --route --preview   # M2: 5 m routing grid + demo route
  python scripts/precompute_layers.py --layers ortho                    # M3: 25 cm ortho -> 1 m U-Net input
  python scripts/precompute_layers.py --layers landform --route --preview  # M5: landform map + fused-cost route
  python scripts/precompute_layers.py --layers pred --preview           # M7: U-Net classes + probs, test/val, 5 m
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import binary_dilation
from skimage.measure import block_reduce
from tqdm import tqdm

import torch
from matplotlib.colors import LightSource
from rasterio.transform import Affine, rowcol

from martianmap.config import load_config, repo_path
from martianmap.models.hazard_unet import load_model, predict_probs
from martianmap.models.landform import CLASSES as LANDFORM_CLASSES
from martianmap.models.landform import UNKNOWN, landform_cell_probs, load_domars, sample_to_grid
from martianmap.terrain.costmap import block_max, block_mean, buffer_impassable, landform_multiplier, slope_cost
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


def ortho_1m(cfg: dict) -> None:
    """25 cm ortho -> exact block average onto the 1 m DTM grid; a block with any nodata (0) becomes 0."""
    p = cfg["paths"]
    out = repo_path(p["ortho_1m"])
    with rasterio.open(repo_path(p["ortho_raw"])) as src, rasterio.open(repo_path(p["dtm_raw"])) as dtm:
        f = round(dtm.res[0] / src.res[0])
        assert src.bounds == dtm.bounds and src.width == dtm.width * f and src.height == dtm.height * f, "ortho/DTM grids differ"
        meta = dict(driver="GTiff", width=dtm.width, height=dtm.height, count=1, dtype="uint8", nodata=0, crs=dtm.crs,
                    transform=dtm.transform, tiled=True, blockxsize=512, blockysize=512, compress="deflate")
        with rasterio.open(out, "w", **meta) as dst:
            for r0 in tqdm(range(0, dtm.height, STRIP_ROWS), desc="ortho -> 1 m"):
                n = min(STRIP_ROWS, dtm.height - r0)
                a = src.read(1, window=Window(0, r0 * f, src.width, n * f)).reshape(n, f, dtm.width, f)
                img = np.rint(a.mean(axis=(1, 3), dtype=np.float32)).astype(np.uint8)
                img[(a == 0).any(axis=(1, 3))] = 0
                dst.write(img, 1, window=Window(0, r0, dtm.width, n))
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


def landform_layer(cfg: dict, preview_png: bool) -> None:
    """DoMars16k sliding-window landform map over the CTX covering the DTM -> landform.npz (+ PNG)."""
    d, p = cfg["domars"], cfg["paths"]
    with rasterio.open(repo_path(p["dtm_raw"])) as dtm, rasterio.open(repo_path(p["ctx_raw"])) as ctx:
        margin = d["patch_size"] * ctx.res[0] / 2  # so cells at the DTM edge still get full windows
        b = dtm.bounds
        win = ctx.window(b.left - margin, b.bottom - margin, b.right + margin, b.top + margin).round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, ctx.width, ctx.height))
        img = ctx.read(1, window=win)
        win_transform = ctx.window_transform(win)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_domars(repo_path(d["weights"]), d["arch"], device)
    probs = landform_cell_probs(net, img, d["patch_size"], d["stride"], device)
    confident = np.nan_to_num(probs).max(0) >= d["min_confidence"]  # NaN (no window) is never confident
    cls = np.where(confident, np.nan_to_num(probs).argmax(0), UNKNOWN).astype(np.uint8)
    cell_transform = win_transform * Affine.scale(d["stride"])
    out = repo_path(d["map"])
    np.savez_compressed(out, probs=probs.astype(np.float16), cls=cls, transform=np.array(cell_transform)[:6],
                        classes=np.array(LANDFORM_CLASSES))
    known = cls[cls != UNKNOWN]
    share = np.bincount(known, minlength=len(LANDFORM_CLASSES)) / known.size
    print(f"landform map {cls.shape} cells of {d['stride'] * ctx.res[0]:g} m, {100 * confident.mean():.0f}% confident "
          f"(>= {d['min_confidence']}); share of confident cells: "
          + ", ".join(f"{LANDFORM_CLASSES[i]} {100 * share[i]:.0f}%" for i in np.argsort(-share) if share[i] > 0.005))
    print(f"wrote {out}")
    if not preview_png:
        return
    cmap = plt.get_cmap("tab20", len(LANDFORM_CLASSES))
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    ax[0].imshow(img, cmap="gray")
    k = d["stride"]
    ax[1].imshow(img, cmap="gray")
    ax[1].imshow(np.ma.masked_equal(cls, UNKNOWN), cmap=cmap, vmin=-0.5, vmax=len(LANDFORM_CLASSES) - 0.5, alpha=0.55,
                 extent=(0, cls.shape[1] * k, cls.shape[0] * k, 0), interpolation="nearest")
    present = [i for i in range(len(LANDFORM_CLASSES)) if share[i] > 0]
    ax[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in present],
                 labels=[LANDFORM_CLASSES[i] for i in present], loc="upper left", fontsize=9, ncol=2)
    ax[0].set_title("CTX (6 m)")
    ax[1].set_title(f"DoMars16k landform, {k * ctx.res[0]:g} m cells (uncoloured = confidence < {d['min_confidence']})")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    png = repo_path(p["previews"]) / "landform.png"
    fig.savefig(png, dpi=90)
    print(f"wrote {png}")


def demo_route(cfg: dict, preview_png: bool) -> None:
    """Routes between the config demo endpoints: slope-only, slope + landform (if the landform map exists),
    and the straight line. All stats are measured on the full (slope + landform) cost."""
    g = np.load(repo_path(cfg["paths"]["route_grid"]))
    slope, dem, cell = g["slope"], g["dem"], float(g["cell_m"])
    transform = Affine(*g["transform"])
    c = cfg["cost"]
    buffer = round(c["hazard_buffer_m"] / cell)
    slope_only = slope_cost(slope, c["slope_scale_deg"], c["max_slope_deg"], buffer)
    lf_path = repo_path(cfg["domars"]["map"])
    lf = None
    if lf_path.exists():
        m = np.load(lf_path)
        lf = sample_to_grid(m["cls"], Affine(*m["transform"]), transform, slope.shape)
        mult = landform_multiplier(lf, LANDFORM_CLASSES, c["landform_multiplier"])
        cost = buffer_impassable(slope_cost(slope, c["slope_scale_deg"], c["max_slope_deg"]) * mult, buffer)
    else:
        print(f"no landform map at {lf_path}: slope-only cost")
        cost = slope_only
    pairs = [("demo", cfg["route"], "route.png")]
    if "landform_demo" in cfg["route"]:
        pairs.append(("landform demo", cfg["route"]["landform_demo"], "route_landform.png"))
    shade = LightSource(azdeg=315, altdeg=45).hillshade(np.nan_to_num(dem, nan=np.nanmean(dem)), dx=cell, dy=cell)
    shade[np.isnan(dem)] = np.nan
    for title, ends, png in pairs:
        start, goal = (tuple(int(v) for v in rowcol(transform, *ends[k])) for k in ("start_xy", "goal_xy"))
        for name, (r, col) in (("start", start), ("goal", goal)):
            if not (0 <= r < cost.shape[0] and 0 <= col < cost.shape[1]):
                raise SystemExit(f"{title} {name} {ends[name + '_xy']} is outside the routing grid")
        paths = {"slope+landform": least_cost_route(cost, start, goal)} if lf is not None else {}
        paths |= {"slope only": least_cost_route(slope_only, start, goal), "straight": straight_line(start, goal)}
        caution = cfg["hazard"]["thresholds_deg"][0]
        stats = {k: path_stats(v, slope, dem, cost, cell, caution) for k, v in paths.items()}
        first = next(iter(stats.values()))
        print(f"\n== {title} route ==\n{'':15s}" + "".join(f"{k:>22s}" for k in first))
        for k, st in stats.items():
            print(f"{k:15s}" + "".join(f"{v:22.1f}" for v in st.values()))
        if lf is not None:
            print("% of route per landform (unknown = below confidence gate or outside the map):")
            for k, v in paths.items():
                ids, n = np.unique(lf[v[:, 0], v[:, 1]], return_counts=True)
                names = ["unknown" if i == UNKNOWN else LANDFORM_CLASSES[i] for i in ids]
                print(f"  {k:15s}" + ", ".join(f"{nm} {100 * c_ / len(v):.0f}%" for nm, c_ in sorted(zip(names, n), key=lambda t: -t[1])))
        if not preview_png:
            continue
        out = repo_path(cfg["paths"]["previews"]) / png
        blocked = np.ma.masked_where(np.isfinite(cost), np.ones_like(cost))
        (r0, c0), (r1, c1) = np.vstack(list(paths.values())).min(0) - 60, np.vstack(list(paths.values())).max(0) + 60
        fig, ax = plt.subplots(1, 2, figsize=(18, 9))
        for a, zoom in zip(ax, (False, True)):
            a.imshow(shade, cmap="gray")
            a.imshow(blocked, cmap="autumn", alpha=0.6, interpolation="nearest")
            for (k, pth), style in zip(paths.items(), ("r-", "m-", "c--") if len(paths) == 3 else ("r-", "c--")):
                a.plot(pth[:, 1], pth[:, 0], style, lw=1.5, label=f"{k}: {stats[k]['length_m']:.0f} m")
            a.plot(*start[::-1], "go", ms=8)
            a.plot(*goal[::-1], "b*", ms=12)
            if zoom:
                a.set_xlim(max(c0, 0), c1)
                a.set_ylim(r1, max(r0, 0))
            a.axis("off")
        ax[0].set_title(f"{title}: {cell:g} m grid; red/orange = impassable (slope >= {c['max_slope_deg']:g} deg, cliff, or no data)")
        ax[1].legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print(f"wrote {out}")


def predicted_hazard(cfg: dict, preview_png: bool) -> None:
    """Hazard U-Net over the test and val regions (never trained on), per 5 m routing cell: max class and
    max softmax probability per class (block max, like true slope) -> one .npz per region."""
    p, t, d = cfg["paths"], cfg["tiles"], cfg["dtm_free"]
    f = round(cfg["grid"]["route_cell_size_m"] / cfg["grid"]["dtm_pixel_size_m"])
    ctx = d["context_px"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(repo_path(d["ckpt"]), device)
    for split, out in d["pred_maps"].items():
        r0, r1, c0, c1 = (v * t["block_m"] for v in t[f"{split}_blocks"])
        with rasterio.open(repo_path(p["ortho_1m"])) as o:
            r1, c1 = (min(r1, o.height) // f) * f, (min(c1, o.width) // f) * f
            assert r0 % f == 0 and c0 % f == 0, f"{split} region must start on a routing-cell boundary"
            cr0, cc0, cr1, cc1 = max(r0 - ctx, 0), max(c0 - ctx, 0), min(r1 + ctx, o.height), min(c1 + ctx, o.width)
            img = o.read(1, window=Window(cc0, cr0, cc1 - cc0, cr1 - cr0))
            transform = o.transform
        probs = predict_probs(model, img / 255, device=device)
        cls = probs.argmax(0).astype(np.uint8)
        # the model calls the ortho's no-data edge a scarp: drop predictions near it
        masked = binary_dilation(img == 0, iterations=d["nodata_margin_px"])
        cls[masked] = IGNORE
        probs[:, masked] = np.nan
        inner = (slice(r0 - cr0, r1 - cr0), slice(c0 - cc0, c1 - cc0))
        cells = block_reduce(cls[inner], (f, f), np.max)  # same conservative rule as true slope (block max); 255 wins
        cell_probs = block_reduce(probs[(slice(None), *inner)], (1, f, f), np.max)  # NaN (masked) wins
        out = repo_path(out)
        np.savez_compressed(out, cls=cells, probs=cell_probs.astype(np.float16), row0=r0 // f, col0=c0 // f,
                            transform=np.array(transform * Affine.translation(c0, r0) * Affine.scale(f))[:6])
        share = np.bincount(cells[cells != IGNORE], minlength=3) / max((cells != IGNORE).sum(), 1)
        print(f"{split}: wrote {out}  {cells.shape} cells, {100 * (cells != IGNORE).mean():.0f}% predicted; "
              f"safe {100 * share[0]:.1f}% caution {100 * share[1]:.1f}% hazard {100 * share[2]:.1f}% (5 m cell max)")
        if preview_png:
            png = repo_path(p["previews"]) / f"pred_hazard_{split}.png"
            plt.imsave(png, np.where(cells == IGNORE, 3, cells), cmap=matplotlib.colors.ListedColormap(
                ["#2a9d8f", "#e9c46a", "#e63946", "#000000"]), vmin=0, vmax=3)
            print(f"wrote {png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", nargs="*", default=[], choices=["slope", "hazard", "cost", "ortho", "landform", "pred"])
    ap.add_argument("--route", action="store_true", help="route the config demo pairs (route.start_xy/goal_xy and route.landform_demo)")
    ap.add_argument("--preview", action="store_true", help="write downsampled PNGs to data/processed/previews")
    args = ap.parse_args()
    cfg = load_config()
    if {"slope", "hazard"} & set(args.layers):  # both come out of the same pass over the DTM
        slope_and_hazard(cfg)
        if args.preview:
            preview(cfg)
    if "ortho" in args.layers:
        ortho_1m(cfg)
    if "landform" in args.layers:
        landform_layer(cfg, args.preview)
    if "pred" in args.layers:
        predicted_hazard(cfg, args.preview)
    if "cost" in args.layers:  # cost itself is cheap and computed at route time from the grid's slope
        route_grid(cfg)
    if args.route:
        demo_route(cfg, args.preview)


if __name__ == "__main__":
    main()
