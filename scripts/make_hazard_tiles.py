"""Cut the 1 m ortho + hazard labels into tiles with spatial block splits -> tiles.csv + split map PNG.

Usage: python scripts/make_hazard_tiles.py
Tiles are an index (pixel row/col); the Dataset reads windows from ortho_1m.tif / hazard_labels.tif.
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Rectangle
from rasterio.transform import rowcol

from martianmap.config import load_config, repo_path
from martianmap.data.tiles import DROPPED, SPLIT_NAMES, assign_split, block_split_map, tile_origins

COLORS = {0: "tab:green", 1: "tab:blue", 2: "tab:red"}


def main() -> None:
    cfg = load_config()
    p, t = cfg["paths"], cfg["tiles"]
    with rasterio.open(repo_path(p["hazard_labels"])) as l, rasterio.open(repo_path(p["ortho_1m"])) as o:
        assert l.transform == o.transform and l.shape == o.shape, "label/ortho grids differ"
        labels, img, transform, px = l.read(1), o.read(1), l.transform, l.res[0]
    labels[img == 0] = 255  # no image -> nothing to learn from
    tile, block, buffer = t["size_px"], round(t["block_m"] / px), round(t["buffer_m"] / px)
    split_map = block_split_map(-(-labels.shape[0] // block), -(-labels.shape[1] // block), t["val_blocks"], t["test_blocks"])

    rows, counts = [], {s: np.zeros(3, np.int64) for s in SPLIT_NAMES}
    n_dropped_buffer = n_dropped_valid = 0
    for r, c in tile_origins(*labels.shape, tile):
        lab = labels[r:r + tile, c:c + tile]
        frac = np.bincount(lab.ravel(), minlength=256)
        valid = frac[:3].sum() / lab.size
        if valid < t["min_valid_frac"]:
            n_dropped_valid += 1
            continue
        split = assign_split(r, c, tile, buffer, block, split_map)
        if split == DROPPED:
            n_dropped_buffer += 1
            continue
        counts[split] += frac[:3]
        rows.append(dict(row=r, col=c, split=SPLIT_NAMES[split], valid=round(valid, 3),
                         **{k: round(frac[i] / frac[:3].sum(), 4) for i, k in enumerate(("safe", "caution", "hazard"))}))

    out = repo_path(p["tiles_index"])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    total_px = sum(c.sum() for c in counts.values())
    print(f"{len(rows)} tiles of {tile}x{tile} px kept; dropped {n_dropped_valid} (<{t['min_valid_frac']:.0%} valid), "
          f"{n_dropped_buffer} (within {t['buffer_m']} m of another split)")
    print(f"{'split':6s} {'tiles':>6s} {'% of px':>8s} {'safe':>7s} {'caution':>8s} {'hazard':>7s}")
    for s, name in SPLIT_NAMES.items():
        n, c = sum(r["split"] == name for r in rows), counts[s]
        print(f"{name:6s} {n:6d} {100 * c.sum() / total_px:7.1f}% " + " ".join(f"{100 * x / c.sum():7.2f}%" for x in c))
    print(f"wrote {out}")

    # split map: ortho thumbnail, kept tiles coloured by split, split blocks outlined, demo endpoints
    k = 16
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img[::k, ::k], cmap="gray")
    for r in rows:
        s = {v: i for i, v in SPLIT_NAMES.items()}[r["split"]]
        ax.add_patch(Rectangle((r["col"] / k, r["row"] / k), tile / k, tile / k, fc=COLORS[s], alpha=0.35, ec="none"))
    for s, key in ((1, "val_blocks"), (2, "test_blocks")):
        r0, r1, c0, c1 = t[key]
        ax.add_patch(Rectangle((c0 * block / k, r0 * block / k), (c1 - c0) * block / k, (r1 - r0) * block / k,
                               fill=False, ec=COLORS[s], lw=2))
    for key, m in (("start_xy", "go"), ("goal_xy", "b*")):
        rr, cc = rowcol(transform, *cfg["route"][key])
        ax.plot(cc / k, rr / k, m, ms=10)
    ax.set_title("Tiles: green train / blue val / red test (outlines = split blocks)")
    ax.axis("off")
    fig.tight_layout()
    png = repo_path(p["previews"]) / "splits.png"
    fig.savefig(png, dpi=100)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
