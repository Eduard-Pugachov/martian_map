"""Martian Map: hazard-aware walking routes in Jezero Crater (Gradio, CPU, precomputed layers, A* only).

Run: python app/app.py   (needs app/data/layers.npz from scripts/export_app_data.py)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from rasterio.transform import Affine, rowcol  # noqa: E402

from martianmap.config import load_config, repo_path  # noqa: E402
from martianmap.terrain.compare_routes import build_costs, compare, risk_classes  # noqa: E402
from martianmap.terrain.costmap import landform_multiplier  # noqa: E402
from martianmap.terrain.route import least_cost_route, path_stats  # noqa: E402

CFG = load_config()
B = np.load(repo_path(CFG["app"]["bundle"]))
SLOPE, DEM, PRED = B["slope"], B["dem"], B["pred"]
P_HAZARD = B["p_hazard"].astype(np.float32)
CELL = float(B["cell_m"])
TRANSFORM = Affine(*B["transform"])
LF_CLASSES = tuple(str(c) for c in B["landform_classes"])
T_RISK = CFG["dtm_free"]["hazard_prob_threshold"]
THRESH = CFG["hazard"]["thresholds_deg"]
MULT = landform_multiplier(B["landform"], LF_CLASSES, CFG["cost"]["landform_multiplier"])
COSTS, TRUE_COST = build_costs(SLOPE, PRED, P_HAZARD, MULT, CFG, CELL, [T_RISK])
RISK_KEY = f"P@{T_RISK:g}"

MODE_DTM = "DTM mode (true slope from the elevation model)"
MODE_FREE = "DTM-free mode (U-Net hazard from the image only)"
OVERLAYS = ["None", "True slope hazard (DTM)", "Predicted hazard (U-Net)", "Landform (DoMars16k)"]
NAMES = {"T": "DTM route", RISK_KEY: "DTM-free route (U-Net, risk-aware)", "P": "Plain U-Net route (no risk threshold)"}
COLORS = {"T": (255, 255, 255), RISK_KEY: (0, 200, 255), "P": (255, 0, 200)}
LANDFORM_NAMES = {
    "aec": "aeolian straight (dunes)", "ael": "aeolian curved (dunes)", "cli": "cliff", "cra": "crater",
    "fse": "slope streaks", "fsf": "channel", "fsg": "gullies", "fss": "mass wasting", "mix": "mixed terrain",
    "rid": "ridge", "rou": "rough terrain", "sfe": "mounds", "sfx": "crater field", "smo": "smooth terrain",
    "tex": "textured terrain",
}


def _hillshade() -> np.ndarray:
    z = np.nan_to_num(DEM, nan=float(np.nanmean(DEM)))
    hs = LightSource(azdeg=315, altdeg=45).hillshade(z, dx=CELL, dy=CELL)
    hs[np.isnan(DEM)] = 0
    return hs


def _stretch(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img[img > 0], [1, 99])
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _class_overlay(cls: np.ndarray) -> np.ndarray:
    """RGBA: caution yellow, hazard red, safe and no-data transparent."""
    rgba = np.zeros((*cls.shape, 4), np.float32)
    rgba[cls == 1] = (0.95, 0.8, 0.2, 0.45)
    rgba[cls == 2] = (0.9, 0.1, 0.1, 0.7)
    return rgba


def _landform_overlay() -> np.ndarray:
    cmap = colormaps["tab20"]
    lf = B["landform"]
    rgba = np.zeros((*lf.shape, 4), np.float32)
    for i in np.unique(lf[lf != 255]):
        rgba[lf == i] = (*cmap(int(i) % 20)[:3], 0.45)
    return rgba


with np.errstate(invalid="ignore"):
    TRUE_CLS = np.where(np.isfinite(SLOPE), np.digitize(SLOPE, THRESH), 255).astype(np.uint8)
BASE = {MODE_DTM: _hillshade(), MODE_FREE: _stretch(B["ortho"])}
OVERLAY = {
    OVERLAYS[1]: _class_overlay(TRUE_CLS),
    OVERLAYS[2]: _class_overlay(risk_classes(PRED, P_HAZARD, T_RISK)),
    OVERLAYS[3]: _landform_overlay(),
}


def landform_legend() -> str:
    lf = B["landform"]
    present = [LF_CLASSES[i] for i in np.unique(lf[lf != 255])]
    cmap = colormaps["tab20"]
    items = []
    for code in present:
        r, g, b = (int(255 * v) for v in cmap(LF_CLASSES.index(code) % 20)[:3])
        mult = CFG["cost"]["landform_multiplier"].get(code, 1.0)
        items.append(f'<span style="color:rgb({r},{g},{b})">&#9632;</span> {LANDFORM_NAMES[code]} '
                     f'(cost x{"impassable" if mult is None else f"{mult:g}"})')
    return "Landform classes in this region: " + ", ".join(items) + ". Uncoloured = low confidence (cost x1)."


def plan(mode: str, start: tuple[int, int], goal: tuple[int, int], show_plain: bool) -> tuple[dict, pd.DataFrame]:
    """Routes for the mode, each scored on the true DTM terrain. Raises ValueError for blocked endpoints."""
    keys = ["T"] if mode == MODE_DTM else [RISK_KEY, *(["P"] if show_plain else []), "T"]
    paths = {k: least_cost_route(COSTS[k], start, goal) for k in keys}
    radius = round(CFG["dtm_free"]["overlap_radius_m"] / CELL)
    rows = []
    for k, p in paths.items():
        c = compare(p, paths["T"], TRUE_COST, SLOPE, DEM, CELL, THRESH, radius)
        climb = path_stats(p, SLOPE, DEM, TRUE_COST, CELL, THRESH[0])["climb_m"]
        extra = c["extra_cost_pct"]
        rows.append({
            "Route": NAMES[k],
            "Length (m)": round(c["length_m"]),
            "Climb (m)": round(climb),
            "Max true slope (deg)": round(c["max_slope_deg"], 1),
            "True hazard cells crossed": c["hazard_cells"],
            "Extra true cost vs DTM route": "crosses impassable terrain" if not np.isfinite(extra) else f"{extra:+.1f}%",
        })
    return paths, pd.DataFrame(rows)


def render(mode: str, overlay: str, pts: dict, paths: dict) -> np.ndarray:
    img = np.repeat(BASE[mode][..., None], 3, axis=2)
    if overlay in OVERLAY:
        o = OVERLAY[overlay]
        img = img * (1 - o[..., 3:]) + o[..., :3] * o[..., 3:]
    img = (img * 255).astype(np.uint8)
    h, w = img.shape[:2]
    for k in ("T", "P", RISK_KEY):  # reference first, chosen route drawn on top
        if k in paths:
            p = paths[k]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    img[np.clip(p[:, 0] + dr, 0, h - 1), np.clip(p[:, 1] + dc, 0, w - 1)] = COLORS[k]
    for name, color in (("start", (40, 220, 40)), ("goal", (40, 90, 255))):
        if pts.get(name):
            r, c = pts[name]
            img[max(r - 6, 0) : r + 7, max(c - 6, 0) : c + 7] = (0, 0, 0)
            img[max(r - 4, 0) : r + 5, max(c - 4, 0) : c + 5] = color
    return img


EMPTY = pd.DataFrame(columns=["Route", "Length (m)", "Climb (m)", "Max true slope (deg)", "True hazard cells crossed",
                              "Extra true cost vs DTM route"])


def update(mode: str, overlay: str, pts: dict, show_plain: bool):
    """Re-plan (if both points are set) and redraw."""
    if not (pts.get("start") and pts.get("goal")):
        msg = "Click the map to set a start point." if not pts.get("start") else "Start set. Click the map to set the goal."
        return render(mode, overlay, pts, {}), EMPTY, msg
    try:
        paths, table = plan(mode, pts["start"], pts["goal"], show_plain)
    except ValueError as e:
        return render(mode, overlay, pts, {}), EMPTY, f"No route: {e}. Pick a point off the blocked terrain."
    msg = ("Route planned on the true slope." if mode == MODE_DTM else
           f"Route planned from the image only (U-Net, hazard if p >= {T_RISK:g}); the white DTM route is shown "
           "for reference. All numbers are measured on the true terrain.")
    return render(mode, overlay, pts, paths), table, msg


def on_click(mode: str, overlay: str, pts: dict, show_plain: bool, evt: gr.SelectData):
    x, y = evt.index
    pt = (int(y), int(x))
    pts = {"start": pt, "goal": None} if (not pts.get("start") or pts.get("goal")) else {**pts, "goal": pt}
    return pts, *update(mode, overlay, pts, show_plain)


def demo_pair(mode: str, overlay: str, show_plain: bool):
    pts = {k: tuple(int(v) for v in rowcol(TRANSFORM, *CFG["route"][f"{k}_xy"])) for k in ("start", "goal")}
    return pts, *update(mode, overlay, pts, show_plain)


def clear(mode: str, overlay: str, show_plain: bool):
    pts = {"start": None, "goal": None}
    return pts, *update(mode, overlay, pts, show_plain)


HOW_IT_WORKS = f"""
## What this is
A route planner for astronauts walking in Jezero Crater, Mars (the Perseverance landing site). It finds the
cheapest walking route between two points while avoiding steep terrain, and it can do this **without an
elevation model**, from a single orbital image, using a neural network.

The map covers a 6 x 6 km region that the hazard model never saw during training, at 5 m per cell.

## Data
- **HiRISE DTM** (1 m elevation model) and **HiRISE orthoimage** of Jezero (USGS / NASA).
- **CTX** 6 m image mosaic for landforms.
- **DoMars16k** landform dataset and pretrained classifier (Wilhelm et al.).

## How it works
1. **Slope hazard labels.** Slope from the DTM (about a 3 m baseline): safe below {THRESH[0]:g} deg,
   caution {THRESH[0]:g}-{THRESH[1]:g} deg, hazard at {THRESH[1]:g} deg or more.
2. **Hazard U-Net (main model).** A ResNet-34 U-Net predicts safe / caution / hazard for every pixel from the
   orthoimage alone. Trained on spatially separate blocks; this region is held out.
3. **Landform map.** The DoMars16k ResNet-50 classifies 1.2 km CTX windows into 15 landforms (dunes, craters,
   cliffs, ...). Each landform multiplies the walking cost (for example dunes x4).
4. **Routing.** Least-cost path (8-connected) on a 5 m grid.
   - DTM mode: cost = (1 + (slope / 10 deg)^2) x landform, impassable at {CFG['cost']['max_slope_deg']:g} deg, with a
     {CFG['cost']['hazard_buffer_m']:g} m safety buffer.
   - DTM-free mode: cost per predicted class (safe 1, caution 3, hazard 50) x landform. **Risk-aware rule:** a
     cell counts as hazard whenever the U-Net gives it a hazard probability of {T_RISK:g} or more, even if another
     class is more likely. The threshold was chosen on a separate validation region, then tested once here.

## How well it works
**Hazard U-Net, pixel level, on this held-out region** (mean IoU over the 3 classes):

| Method | mIoU | Safe | Caution | Hazard |
|---|---|---|---|---|
| U-Net | 0.547 | 0.86 | 0.42 | 0.36 |
| Image-gradient baseline | 0.288 | | | |
| Always "safe" | 0.283 | | | |

**Routes, 200 random start/goal pairs (2-6 km) in this region**, all scored on the true terrain:

| Route | Pairs crossing a true hazard (95% CI) | Median extra cost vs DTM route |
|---|---|---|
| Oracle (true slope cut into the 3 classes) | 0% (0-1.9%) | 10.1% |
| **DTM-free, risk-aware U-Net** | **5.5% (3.1-9.6%)** | 15.2% |
| Plain U-Net (most likely class) | 21.5% (16.4-27.7%) | 15.8% |
| Landform only, no hazard information | 77.5% (71.2-82.7%) | 13.5% |

On the demo route (landing site to the delta), the plain U-Net route walks down a 25-34 deg scarp face the model
called "caution"; the risk-aware route avoids it (max true slope 24.0 deg) for 20.5% extra cost.

## Limitations
- Single-image slope depends on the sun: **west-facing slopes are much harder** (hazard IoU 0.14 vs 0.35-0.42).
- On the validation region (steep northern scarps) DTM-free routes crossed a hazard on about half the pairs, and
  the risk threshold gave little gain there. The 5.5% above is for this region, not a guarantee elsewhere.
- Cost weights, slope thresholds and landform multipliers are reasoned assumptions, not calibrated against
  astronaut data. The landform classifier was trained on other parts of Mars; some classes (for example "mass
  wasting" on the flat floor) are likely wrong here.
- 5 m cells: features smaller than a few metres (boulders, small ripples) are not resolved.
- A planning aid for exploring ideas, not a safety-certified tool.
"""

with gr.Blocks(title="Martian Map: Jezero") as demo:
    gr.Markdown("# Martian Map: hazard-aware walking routes in Jezero Crater\n"
                "Click the map to set a start and a goal, or load the demo route. Switch to DTM-free mode to plan "
                "from the image alone and compare it with the route from the elevation model.")
    pts = gr.State({"start": None, "goal": None})
    with gr.Tab("Mission Map"):
        with gr.Row():
            with gr.Column(scale=3):
                img = gr.Image(type="numpy", interactive=False, show_label=False, height=720)
            with gr.Column(scale=1):
                mode = gr.Radio([MODE_DTM, MODE_FREE], value=MODE_DTM, label="Mode")
                overlay = gr.Radio(OVERLAYS, value=OVERLAYS[0], label="Map overlay")
                show_plain = gr.Checkbox(False, label="DTM-free: also show the plain U-Net route (no risk threshold)")
                with gr.Row():
                    demo_btn = gr.Button("Demo route", variant="primary")
                    clear_btn = gr.Button("Clear")
                status = gr.Markdown()
                gr.Markdown("**Legend.** Routes: white = DTM route, cyan = DTM-free route, magenta = plain U-Net "
                            "route. Green square = start, blue square = goal. Hazard overlays: yellow = caution "
                            f"({THRESH[0]:g}-{THRESH[1]:g} deg), red = hazard (>= {THRESH[1]:g} deg). DTM mode shows "
                            "a hillshade of the elevation model; DTM-free mode shows the orthoimage the U-Net sees.")
                gr.Markdown(landform_legend())
        table = gr.Dataframe(value=EMPTY, interactive=False, label="Route statistics (measured on the true terrain)")
    with gr.Tab("How it works"):
        gr.Markdown(HOW_IT_WORKS)

    outs = [img, table, status]
    for ctl in (mode, overlay, show_plain):
        ctl.change(update, [mode, overlay, pts, show_plain], outs)
    img.select(on_click, [mode, overlay, pts, show_plain], [pts, *outs])
    demo_btn.click(demo_pair, [mode, overlay, show_plain], [pts, *outs])
    clear_btn.click(clear, [mode, overlay, show_plain], [pts, *outs])
    demo.load(update, [mode, overlay, pts, show_plain], outs)

if __name__ == "__main__":
    demo.launch()
