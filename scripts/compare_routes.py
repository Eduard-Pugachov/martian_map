"""DTM-free mode comparison (PLAN §3E): routes T / O / P (+ L baseline) in the test region.

  T  true DTM slope cost (reference)          O  oracle: true slope cut into the 3 classes
  P  hazard U-Net predicted classes           L  baseline: landform only, no hazard information
All share the landform multiplier and the impassable buffer; every route is scored on the true
cost (DTM slope x landform, without the buffer).

Usage: python scripts/compare_routes.py [--preview]
Needs: precompute_layers.py --layers cost landform pred
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, ListedColormap
from rasterio.transform import Affine, rowcol
from tqdm import tqdm

from martianmap.config import load_config, repo_path
from martianmap.models.landform import CLASSES as LANDFORM_CLASSES
from martianmap.models.landform import sample_to_grid
from martianmap.terrain.compare_routes import class_cost, compare, wilson_ci95
from martianmap.terrain.costmap import buffer_impassable, landform_multiplier, slope_cost
from martianmap.terrain.hazard_labels import IGNORE
from martianmap.terrain.route import least_cost_route

ROUTES = ("O", "P", "L")


def cost_maps(cfg: dict) -> dict:
    """Slope, DEM and the T/O/P/L cost maps, cropped to the predicted (test) region."""
    g = np.load(repo_path(cfg["paths"]["route_grid"]))
    pr = np.load(repo_path(cfg["dtm_free"]["pred_map"]))
    pred, r0, c0 = pr["cls"], int(pr["row0"]), int(pr["col0"])
    win = (slice(r0, r0 + pred.shape[0]), slice(c0, c0 + pred.shape[1]))
    slope, dem, cell = g["slope"][win], g["dem"][win], float(g["cell_m"])
    transform = Affine(*pr["transform"])
    m = np.load(repo_path(cfg["domars"]["map"]))
    lf = sample_to_grid(m["cls"], Affine(*m["transform"]), transform, slope.shape)
    c, d = cfg["cost"], cfg["dtm_free"]
    mult = landform_multiplier(lf, LANDFORM_CLASSES, c["landform_multiplier"])
    buf = round(c["hazard_buffer_m"] / cell)
    with np.errstate(invalid="ignore"):
        oracle = np.where(np.isfinite(slope), np.digitize(slope, cfg["hazard"]["thresholds_deg"]), IGNORE).astype(np.uint8)
    true = slope_cost(slope, c["slope_scale_deg"], c["max_slope_deg"]) * mult
    costs = {
        "T": buffer_impassable(true, buf),
        "O": buffer_impassable(class_cost(oracle, d["class_costs"]) * mult, buf),
        "P": buffer_impassable(class_cost(pred, d["class_costs"]) * mult, buf),
        "L": buffer_impassable(np.where(pred == IGNORE, np.inf, 1.0) * mult, buf),
    }
    # score on the unbuffered true cost: the buffer is a planning margin, not ground truth
    return {"slope": slope, "dem": dem, "cell": cell, "transform": transform, "pred": pred, "costs": costs,
            "true": true}


def random_pairs(costs: dict, n: int, dist_m: list[float], cell: float, seed: int) -> list[tuple]:
    """Fixed-seed start/goal cell pairs, both ends passable in every cost map, dist_m apart."""
    ok = np.argwhere(np.logical_and.reduce([np.isfinite(v) for v in costs.values()]))
    rng, pairs = np.random.default_rng(seed), []
    while len(pairs) < n:
        a, b = ok[rng.integers(len(ok))], ok[rng.integers(len(ok))]
        if dist_m[0] <= np.hypot(*(a - b)) * cell <= dist_m[1]:
            pairs.append((tuple(int(v) for v in a), tuple(int(v) for v in b)))
    return pairs


def run_pair(m: dict, start: tuple, goal: tuple, cfg: dict) -> tuple[dict, dict]:
    paths = {k: least_cost_route(m["costs"][k], start, goal) for k in ("T", *ROUTES)}
    radius = round(cfg["dtm_free"]["overlap_radius_m"] / m["cell"])
    args = (m["true"], m["slope"], m["dem"], m["cell"], cfg["hazard"]["thresholds_deg"], radius)
    return paths, {k: compare(paths[k], paths["T"], *args) for k in ("T", *ROUTES)}


def summary(rows: list[dict]) -> dict:
    """Per route: median overlap, % of pairs that cross a true hazard, extra cost over pairs that stay passable."""
    out = {}
    for k in ROUTES:
        st = [r[k] for r in rows]
        finite = [s["extra_cost_pct"] for s in st if np.isfinite(s["extra_cost_pct"])]
        crossing = int(sum(s["hazard_cells"] > 0 for s in st))
        lo, hi = wilson_ci95(crossing, len(st))
        out[k] = {
            "median_overlap_pct": float(np.median([s["overlap_pct"] for s in st])),
            "pct_pairs_crossing_hazard": float(100 * crossing / len(st)),
            "crossing_ci95_lo": lo,
            "crossing_ci95_hi": hi,
            "mean_hazard_cells": float(np.mean([s["hazard_cells"] for s in st])),
            "pct_pairs_truly_passable": float(100 * len(finite) / len(st)),
            "median_extra_cost_pct_if_passable": float(np.median(finite)) if finite else float("nan"),
            "median_length_diff_m": float(np.median([s["length_diff_m"] for s in st])),
        }
    return out


def preview(m: dict, paths: dict, start: tuple, goal: tuple, out) -> None:
    shade = LightSource(azdeg=315, altdeg=45).hillshade(np.nan_to_num(m["dem"], nan=np.nanmean(m["dem"])),
                                                       dx=m["cell"], dy=m["cell"])
    pts = np.vstack(list(paths.values()))
    (r0, c0), (r1, c1) = pts.min(0) - 60, pts.max(0) + 60
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    ax[0].imshow(shade, cmap="gray")
    ax[0].imshow(np.ma.masked_where(np.isfinite(m["costs"]["T"]), np.ones_like(shade)), cmap="autumn", alpha=0.6)
    ax[0].set_title("true DTM: red = impassable (slope >= 25 deg + buffer)")
    pred = np.where(m["pred"] == IGNORE, 3, m["pred"])
    ax[1].imshow(pred, cmap=ListedColormap(["#2a9d8f", "#e9c46a", "#e63946", "#000000"]), vmin=0, vmax=3,
                 interpolation="nearest")
    ax[1].set_title("U-Net predicted hazard (green safe / yellow caution / red hazard)")
    for a in ax:
        for (k, p), style in zip(paths.items(), ("k-", "b--", "m-", "c:")):
            a.plot(p[:, 1], p[:, 0], style, lw=1.6, label=k)
        a.plot(start[1], start[0], "go", ms=8)
        a.plot(goal[1], goal[0], "b*", ms=12)
        a.set_xlim(max(c0, 0), c1)
        a.set_ylim(r1, max(r0, 0))
        a.axis("off")
    ax[0].legend(loc="lower left", title="T true / O oracle / P predicted / L landform only")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true", help="write previews/route_compare.png for the demo pair")
    args = ap.parse_args()
    cfg = load_config()
    d = cfg["dtm_free"]
    m = cost_maps(cfg)
    shape = m["slope"].shape
    demo = tuple(tuple(int(v) for v in rowcol(m["transform"], *cfg["route"][k])) for k in ("start_xy", "goal_xy"))
    if not all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in demo):
        raise SystemExit("demo route endpoints are outside the predicted (test) region")

    paths, demo_stats = run_pair(m, *demo, cfg)
    cols = list(demo_stats["T"])
    print(f"\n== demo pair (all scored on the true DTM cost) ==\n{'':4s}" + "".join(f"{c:>16s}" for c in cols))
    for k, st in demo_stats.items():
        print(f"{k:4s}" + "".join(f"{v:16.1f}" for v in st.values()))

    rows = []
    for start, goal in tqdm(random_pairs(m["costs"], d["n_random_pairs"], d["pair_dist_m"], m["cell"], d["seed"]),
                            desc="random pairs"):
        rows.append(run_pair(m, start, goal, cfg)[1])
    summ = summary(rows)
    print(f"\n== {len(rows)} fixed random pairs in the test region ({d['pair_dist_m'][0]:g}-{d['pair_dist_m'][1]:g} m) ==")
    keys = list(summ["P"])
    print(f"{'':4s}" + "".join(f"{k[:22]:>24s}" for k in keys))
    for k, st in summ.items():
        print(f"{k:4s}" + "".join(f"{v:24.1f}" for v in st.values()))

    out = repo_path(d["results"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"demo": demo_stats, "pairs": rows, "summary": summ}, indent=2))
    print(f"wrote {out}")
    if args.preview:
        preview(m, paths, *demo, repo_path(cfg["paths"]["previews"]) / "route_compare.png")


if __name__ == "__main__":
    main()
