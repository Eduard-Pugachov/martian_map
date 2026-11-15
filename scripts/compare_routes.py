"""DTM-free mode comparison (PLAN §3E): routes T / O / P (+ L baseline) in the test region.

  T  true DTM slope cost (reference)          O  oracle: true slope cut into the 3 classes
  P  hazard U-Net predicted classes           L  baseline: landform only, no hazard information
  P@t  risk-aware P: hazard wherever the cell's U-Net p_hazard >= t
All share the landform multiplier and the impassable buffer; every route is scored on the true
cost (DTM slope x landform, without the buffer).

Usage:
  python scripts/compare_routes.py --sweep       # val pairs only: sweep t, plot, recommend t for config
  python scripts/compare_routes.py [--preview]   # test pairs + demo pair, with config hazard_prob_threshold
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
from martianmap.terrain.compare_routes import build_costs, compare, wilson_ci95
from martianmap.terrain.costmap import landform_multiplier
from martianmap.terrain.hazard_labels import IGNORE
from martianmap.terrain.route import least_cost_route

def cost_maps(cfg: dict, split: str, thresholds: list[float]) -> dict:
    """Slope, DEM and the T/O/P/L (+ P@t per threshold) cost maps, cropped to the split's predicted region."""
    g = np.load(repo_path(cfg["paths"]["route_grid"]))
    pr = np.load(repo_path(cfg["dtm_free"]["pred_maps"][split]))
    pred, r0, c0 = pr["cls"], int(pr["row0"]), int(pr["col0"])
    win = (slice(r0, r0 + pred.shape[0]), slice(c0, c0 + pred.shape[1]))
    slope, dem, cell = g["slope"][win], g["dem"][win], float(g["cell_m"])
    transform = Affine(*pr["transform"])
    m = np.load(repo_path(cfg["domars"]["map"]))
    lf = sample_to_grid(m["cls"], Affine(*m["transform"]), transform, slope.shape)
    mult = landform_multiplier(lf, LANDFORM_CLASSES, cfg["cost"]["landform_multiplier"])
    costs, true = build_costs(slope, pred, pr["probs"][2].astype(np.float32), mult, cfg, cell, thresholds)
    # score on the unbuffered true cost: the buffer is a planning margin, not ground truth
    return {"slope": slope, "dem": dem, "cell": cell, "transform": transform, "pred": pred, "costs": costs,
            "true": true}


def random_pairs(costs: dict, n: int, dist_m: list[float], cell: float, seed: int,
                 max_draws: int = 200_000) -> list[tuple]:
    """Fixed-seed start/goal cell pairs, both ends passable in every cost map, dist_m apart.
    Returns fewer than n if the region can't fit them within max_draws."""
    ok = np.argwhere(np.logical_and.reduce([np.isfinite(v) for v in costs.values()]))
    rng, pairs = np.random.default_rng(seed), []
    for _ in range(max_draws):
        if len(pairs) == n:
            break
        a, b = ok[rng.integers(len(ok))], ok[rng.integers(len(ok))]
        if dist_m[0] <= np.hypot(*(a - b)) * cell <= dist_m[1]:
            pairs.append((tuple(int(v) for v in a), tuple(int(v) for v in b)))
    return pairs


def run_pair(m: dict, start: tuple, goal: tuple, cfg: dict) -> tuple[dict, dict]:
    paths = {k: least_cost_route(c, start, goal) for k, c in m["costs"].items()}
    radius = round(cfg["dtm_free"]["overlap_radius_m"] / m["cell"])
    args = (m["true"], m["slope"], m["dem"], m["cell"], cfg["hazard"]["thresholds_deg"], radius)
    return paths, {k: compare(v, paths["T"], *args) for k, v in paths.items()}


def run_pairs(m: dict, pairs: list[tuple], cfg: dict, desc: str) -> list[dict]:
    """Stats for every pair; skips pairs whose ends aren't connected (unreachable even on the true map)."""
    rows = []
    for start, goal in tqdm(pairs, desc=desc):
        try:
            rows.append(run_pair(m, start, goal, cfg)[1])
        except ValueError:
            pass
    return rows


def print_summary(summ: dict) -> None:
    keys = list(next(iter(summ.values())))
    print(f"{'':6s}" + "".join(f"{k[:22]:>24s}" for k in keys))
    for k, st in summ.items():
        print(f"{k:6s}" + "".join(f"{v:24.1f}" for v in st.values()))


def summary(rows: list[dict]) -> dict:
    """Per route: median overlap, % of pairs that cross a true hazard, extra cost over pairs that stay passable."""
    out = {}
    for k in (k for k in rows[0] if k != "T"):
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
        for (k, p), style in zip(paths.items(), ("k-", "b--", "m-", "c:", "r-", "y-", "w-")):
            a.plot(p[:, 1], p[:, 0], style, lw=1.6, label=k)
        a.plot(start[1], start[0], "go", ms=8)
        a.plot(goal[1], goal[0], "b*", ms=12)
        a.set_xlim(max(c0, 0), c1)
        a.set_ylim(r1, max(r0, 0))
        a.axis("off")
    ax[0].legend(loc="lower left", title="T true / O oracle / P predicted / L landform only / P@t risk-aware")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"wrote {out}")


def sweep(cfg: dict) -> None:
    """Val region only: route every P@t, pick the t with the fewest hazard crossings whose median extra cost
    (over truly passable pairs) stays within the limit. Ties -> lower extra cost."""
    d = cfg["dtm_free"]
    m = cost_maps(cfg, "val", d["t_sweep"])
    pairs = random_pairs(m["costs"], d["n_val_pairs"], d["pair_dist_m"], m["cell"], d["seed"])
    rows = run_pairs(m, pairs, cfg, "val pairs")
    summ = summary(rows)
    print(f"\n== {len(rows)} val pairs ({len(pairs)} drawn; unreachable skipped) ==")
    print_summary(summ)
    limit = d["max_median_extra_cost_pct"]
    ok = [k for k in summ if k.startswith("P@") and summ[k]["median_extra_cost_pct_if_passable"] <= limit]
    best = min(ok, key=lambda k: (summ[k]["pct_pairs_crossing_hazard"], summ[k]["median_extra_cost_pct_if_passable"]),
               default=None)
    print(f"\nrecommended hazard_prob_threshold: {best[2:] if best else 'none meets the extra-cost limit'}")
    out = repo_path(d["sweep_results"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_pairs": len(rows), "summary": summ, "recommended": best}, indent=2))
    print(f"wrote {out}")
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, st in summ.items():
        x, y = st["median_extra_cost_pct_if_passable"], st["pct_pairs_crossing_hazard"]
        lo, hi = st["crossing_ci95_lo"], st["crossing_ci95_hi"]
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", color="C3" if k.startswith("P@") else "0.4", capsize=3)
        ax.annotate(k, (x, y), xytext=(6, 4), textcoords="offset points")
    ax.axvline(limit, ls="--", c="0.6")
    ax.set_xlabel("median extra true cost vs DTM route, passable pairs (%)")
    ax.set_ylabel("pairs crossing a true hazard (%), 95% CI")
    ax.set_title(f"Val region, {len(rows)} pairs: hazard crossings vs extra cost")
    fig.tight_layout()
    png = repo_path(cfg["paths"]["previews"]) / "val_t_sweep.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"wrote {png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true", help="write previews/route_compare.png for the demo pair")
    ap.add_argument("--sweep", action="store_true", help="val only: sweep hazard_prob_threshold candidates")
    args = ap.parse_args()
    cfg = load_config()
    d = cfg["dtm_free"]
    if args.sweep:
        return sweep(cfg)
    t = d["hazard_prob_threshold"]
    m = cost_maps(cfg, "test", [] if t is None else [t])
    shape = m["slope"].shape
    demo = tuple(tuple(int(v) for v in rowcol(m["transform"], *cfg["route"][k])) for k in ("start_xy", "goal_xy"))
    if not all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in demo):
        raise SystemExit("demo route endpoints are outside the predicted (test) region")

    paths, demo_stats = run_pair(m, *demo, cfg)
    cols = list(demo_stats["T"])
    print(f"\n== demo pair (all scored on the true DTM cost) ==\n{'':6s}" + "".join(f"{c:>16s}" for c in cols))
    for k, st in demo_stats.items():
        print(f"{k:6s}" + "".join(f"{v:16.1f}" for v in st.values()))

    pairs = random_pairs(m["costs"], d["n_random_pairs"], d["pair_dist_m"], m["cell"], d["seed"])
    rows = run_pairs(m, pairs, cfg, "test pairs")
    summ = summary(rows)
    print(f"\n== {len(rows)} fixed random pairs in the test region ({d['pair_dist_m'][0]:g}-{d['pair_dist_m'][1]:g} m) ==")
    print_summary(summ)

    out = repo_path(d["results"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"demo": demo_stats, "pairs": rows, "summary": summ}, indent=2))
    print(f"wrote {out}")
    if args.preview:
        preview(m, paths, *demo, repo_path(cfg["paths"]["previews"]) / "route_compare.png")


if __name__ == "__main__":
    main()
