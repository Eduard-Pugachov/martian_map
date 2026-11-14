# Martian Map — ML Terrain Intelligence for Marswalk Planning
**NASA Space Apps 2026 · Challenge: "Interplanetary Survival Guide: Martian Map"**
Hackathon: Sat Nov 14 – Sun Nov 15, 2026 (submit by Nov 15, 11:59 PM local time; check your local event)

> **Status (Nov 13):** coding has started. Env: Python 3.13 venv (`.venv`, CUDA torch). Core data downloaded to `data/raw/`.

---

## 0. The pitch
Elevation models (DTMs) exist only where HiRISE took stereo pairs, which is a small part of Mars. Everywhere else, a
Marswalk planner has images but no slopes. **We train a U-Net to predict slope hazard from a single HiRISE orthoimage**,
add a DoMars16k landform layer on top, and plan hazard-aware routes in **Jezero Crater**. In **DTM-free mode** the route
is planned from the image alone, and we show how close it comes to the route planned with the real DTM.

Why Jezero: the USGS Mars 2020 TRN products give a co-produced HiRISE orthomosaic and DTM (free training labels), and
DoMars16k's authors already ran a Jezero landform analysis, so we have something to check against.

> ⚠️ Before Saturday, read the **official challenge statement + resource list** on spaceappschallenge.org and the
> Nov 13 participant guides. Adjust objectives and wording to match, and use at least one dataset they list.

---

## 1. System overview

```
 HiRISE orthoimage (25 cm → 1 m)        HiRISE DTM (1 m)                 CTX image (~6 m/px)
          │                                    │                                 │
          ▼                                    ▼                                 ▼
 ┌──────────────────────┐          slope (≈3 m baseline)            ┌──────────────────────┐
 │ Hazard U-Net         │◄─labels── → safe / caution / hazard       │ Landform classifier  │
 │ image → safe/caution │  (train)             │                    │ (DoMars16k, 15 cls)  │
 │         /hazard      │                      │                    │ sliding window map   │
 └─────────┬────────────┘                      │                    └──────────┬───────────┘
           │ predicted hazard                  │ true slope                    │ landform
           ▼                                   ▼                               │
   Cost (DTM-free mode)                 Cost (DTM mode)  ◄─────────────────────┤
   = f(pred. hazard, landform)          = f(slope, landform) ◄─────────────────┘
           │                                   │
           ▼                                   ▼
     Route P (predicted)    vs.    Route T (true)   (+ Route O: oracle classes)
           └──────────────┬────────────────────┘
                          ▼
     Comparison: path overlap, extra true cost, true-hazard pixels crossed
                          ▼
     Gradio app (HF Spaces): Mission Map · How it works · [optional] Ground view (AI4Mars)
```

**ML components:**
| Component | Task | Role |
|---|---|---|
| **A. Hazard U-Net** (main) | Image → per-pixel slope-hazard class (3 classes) | Makes hazard maps where no DTM exists; drives DTM-free mode |
| **B. Landform classifier** | 15-class CTX patch classification → sliding-window map | Second cost layer: what the terrain *is* (dunes, cliffs, ...) |
| **G. Ground-view U-Net** (optional) | AI4Mars rover images → soil / bedrock / sand / big rock | Small demo tab only; first thing cut |

---

## 2. Datasets

### Core
| # | Dataset | Use | Access |
|---|---|---|---|
| 1 | **Mars 2020 TRN HiRISE DTM mosaic** (USGS, 1 m/px) | Slope → hazard labels; DTM-mode routing | ~1.8 GB GeoTIFF: `https://planetarymaps.usgs.gov/mosaic/mars2020_trn/HiRISE/JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40.tif` |
| 2 | **Mars 2020 TRN HiRISE orthomosaic** (USGS, 25 cm/px) | U-Net input | **7.4 GB** GeoTIFF (doi:10.5066/P9QJDP48): `https://planetarymaps.usgs.gov/mosaic/mars2020_trn/HiRISE/JEZ_hirise_soc_006_orthoMosaic_25cm_Eqc_latTs0_lon0_first.tif` |
| 3 | **Mars 2020 TRN CTX orthomosaic** (6 m/px) | Input for landform map | 36 MB GeoTIFF (doi:10.5066/P9GV1ND3): `https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0.tif` |
| 4 | **DoMars16k** (TU Dortmund) | Train/fine-tune landform classifier | Zenodo 4291940: `data.zip` (~128 MB) + `models.zip` (pretrained) · github.com/thowilh/geomars |

### Optional
| Dataset | Use | Access |
|---|---|---|
| **Second HiRISE DTM + ortho pair** (another site) | Generalization test for the hazard U-Net (different terrain + sun angle) | USGS STAC: stac.astrogeology.usgs.gov (HiRISE DTMs and orthoimages). Pick one small pair in prep. |
| **AI4Mars** (NASA JPL) | Ground-view demo tab (G) | HF `hassanjbara/AI4MARS` (parquet) · Kaggle `yash92328/ai4mars-terrainaware-autonomous-driving-on-mars` |
| Mars-Bench (NeurIPS 2025) | `mb-boulder_seg` as an extra hazard layer (stretch) | mars-bench.github.io |

**AI4Mars label encoding:** `0=soil, 1=bedrock, 2=sand, 3=big rock, 255=no label`. Merge rover (MXY) and 30 m range
(RNG) masks into 255; loss uses `ignore_index=255`.

---

## 3. Component details

### C. Slope and hazard labels (from the DTM)
- **Slope baseline ≈ 3 m:** Gaussian-smooth the DTM (σ = 1 px), then Horn's 3×3 slope operator at 1 m spacing.
  - **Why not 1 m:** HiRISE DTMs are *posted* at 1 m but stereo matching averages over several pixels, so they only
    resolve features of roughly 3–5 m. A 1 m baseline mostly amplifies stereo noise and mosaic seams.
  - **Why not 10 m+:** that would hide the short, steep scarps a walker actually has to avoid. ~3 m ≈ a few steps.
  - Stretch: compare 2 m vs. 5 m baselines.
- **Classes** (thresholds go in `config.yaml`):
  - `0 = safe`: slope < 10°
  - `1 = caution`: 10–25°
  - `2 = hazard`: > 25° (walking-EVA assumption, tunable)
  - `255 = ignore`: DTM no-data, the DTM ↔ MOLA blend zone ("blend40"), and a buffer around seams between source DTMs
- **First check on Nov 14:** class histogram. If hazard is far below ~1% of pixels, consider hazard > 20°.
- **Roughness (stretch):** std of detrended elevation in a ~5 m window, as an extra label channel or cost term.

### A. Image-only hazard U-Net (main model)
- **Working resolution: 1 m.** Area-average the 25 cm ortho 4×4 onto the DTM grid.
  - Gain: one shared grid, 16× fewer pixels, a 512 px tile = 512 m of context, fast training.
  - Loss: fine texture (boulders, ripple crests) only visible at 25 cm, which may correlate with roughness.
  - Stretch: 0.5 m input with labels upsampled 2×.
- **Co-registration:** ortho and DTM share a projection, but grid origins/extents may differ, so resample the ortho
  onto the DTM grid once (`rasterio` WarpedVRT, average resampling). No feature matching needed.
- **Model:** `smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=3)`, grayscale repeated to 3 channels.
  Tiles 512 (256 if GPU time is tight).
- **Loss:** CrossEntropy with class weights (hazard is rare) + Dice, `ignore_index=255`.
- **Augmentation:** 90° rotations + flips are fine because the label is slope *magnitude*, which doesn't change under
  rotation; they also stop the model relying on one fixed sun direction. Plus brightness / contrast / gamma, slight blur.
- **Training:** AdamW lr 3e-4, cosine, AMP, Colab/Kaggle T4. Smoke-test on a small subset, then a full run.
- **Spatial block split (never random tiles):**
  1. Cut the ortho ∩ DTM overlap into 1 km blocks.
  2. **Test** = one contiguous region that **contains the routing demo AOI**, so DTM-free mode is always shown on
     ground the model never saw. **Val** = a second contiguous region. **Train** = everything else (~70%).
  3. Drop a **300 m buffer** (longer than a tile edge) along every boundary between splits.
  4. Choose regions so each split has some hazard pixels (delta-front scarps, crater rims).
  5. A unit test asserts that no tile lies within the buffer of a tile from another split.
  Exact block coordinates are chosen on Nov 14 once the cropped rasters are inspected, and stored in `config.yaml`.
- **Evaluation (test region only):**
  - mIoU (headline), per-class IoU, confusion matrix.
  - Baseline 1: **always-safe**. High pixel accuracy, poor mIoU — this is why mIoU is the headline.
  - Baseline 2: **image gradient**: Sobel magnitude of the smoothed ortho, thresholded into 3 classes, thresholds fit
    on train to maximize mIoU.
  - Stretch: run on the second STAC site with no fine-tuning.
- **Known limitation: illumination.** Slope from a single image is ambiguous: a slope's brightness depends on the
  sun's direction and elevation, the mosaic combines source images taken under different sun angles, and albedo changes
  (dark sand, bright bedrock) look like shading. The model may also learn landform shortcuts (e.g. "dark ripples =
  caution") rather than geometry. **How we report it:**
  - Per-class IoU on the test region **stratified by DTM aspect** into 4 quadrants (toward / away from / across the sun
    if the source image's sun azimuth is known, otherwise N/E/S/W). A big gap between quadrants = illumination dependence.
  - Second-site result (different sun angle), if done.
  - A failure-case slide, and the limitation stated in the README and the How-it-works tab.

### B. Orbital landform classifier (DoMars16k)
- **Fastest path:** pretrained models from `models.zip` (check format/architecture in the geomars repo). Fallback:
  fine-tune `timm` `resnet50` / `convnext_tiny` on DoMars16k (~20–30 min on a T4 at 224 px).
- **Map:** slide a 200×200 window over the CTX AOI, stride 50–100 px → class probabilities → class raster. Optional 3×3 mode filter.
- **Eval:** test-set accuracy + confusion matrix; qualitative comparison with the Jezero figure in the DoMars16k paper.

### D. Cost map and routes
- **Landform multiplier** (both modes): smooth ×1.0, textured ×1.3, mixed ×1.5, rough ×2.5, crater/crater field ×2.0,
  dunes (aeolian curved/straight) ×4.0, cliff = impassable, gullies/mass wasting ×3.0 (all in `config.yaml`).
- **DTM mode:** slope cost `1 + (slope/10)^2`, impassable if > 25°, × landform multiplier.
- **DTM-free mode:** class cost safe = 1, caution = 3, hazard = 50 (very high, *not* impassable, so a wrong prediction
  can't block every path), × landform multiplier.
- Routing: `skimage.graph.route_through_array(cost, start, end, fully_connected=True, geometric=True)` on a
  2–5 m grid (downsampled for speed).
- **Routing AOI:** ~4×4 km, inside the test region (smaller than the earlier 10×10 km idea because of the split).

### E. DTM-free mode comparison (main U-Net demo)
Three routes between the same start and goal:
- **T**: true continuous DTM slope (reference)
- **O**: "oracle" classes: the thresholded true slope. Separates the error from collapsing slope to 3 classes from the model's error.
- **P**: predicted classes from the hazard U-Net

Reported for **P vs. T** and **O vs. T**, always measured on the *true* cost map:
- **Path overlap:** % of the route's cells within 5 m of route T
- **Extra cost:** true cost of the route ÷ true cost of T − 1
- **Hazard crossings:** number of true-hazard pixels crossed and max true slope along the route
- **Length difference**

Also run on a few fixed start/goal pairs for a summary table in the slides (not just the one the user clicks).

### F. App (Gradio, deployed to Hugging Face Spaces)
- **Tab 1 – Mission Map:** basemap with toggleable overlays (slope, true hazard, predicted hazard, landform, cost).
  Click start and goal (`gr.Image.select`). **DTM-free mode toggle** draws routes T, P (and O) with the comparison table
  and an elevation profile.
- **Tab 2 – How it works:** U-Net metrics vs. baselines, aspect-stratified table, landform metrics, datasets, assumptions, limitations.
- **Tab 3 – Ground view (optional):** AI4Mars U-Net on sample Curiosity images: overlay + class percentages.
- Everything heavy is precomputed (predicted hazard raster, slope, landform, cost maps) and shipped as small `.npy`/PNG.
  The app only runs A* at request time (CPU).

### G. Ground-view U-Net (AI4Mars, optional)
- `smp.Unet("resnet34", imagenet, classes=4)`, 512 px input, CE (`ignore_index=255`, class weights for big rock) + Dice.
  No vertical flips (the horizon matters). Subset training is enough for a demo.
- Eval: mIoU on the expert `test_min3` split. Shown only in Tab 3.

---

## 4. Repo structure (to be created on Nov 14)
```
martian_map/
├── CLAUDE.md
├── PLAN.md
├── README.md                 # judge-facing: what, why, how to run, credits, limitations
├── requirements.txt
├── config.yaml               # AOI, split blocks, slope baseline + thresholds, cost weights, paths
├── data/                     # gitignored
│   ├── raw/                  # big downloads
│   └── processed/            # cropped + resampled rasters, tiles, precomputed layers
├── checkpoints/              # gitignored except final small weights
├── scripts/
│   ├── download_domars.py
│   ├── download_ai4mars.py           # optional
│   ├── precompute_layers.py          # slope, hazard labels, landform map, cost maps
│   ├── make_hazard_tiles.py          # tiles + spatial splits → data/processed/tiles
│   ├── eval_hazard.py                # mIoU, baselines, aspect-stratified table
│   └── compare_routes.py             # T / O / P routes on fixed start-goal pairs
├── src/martianmap/
│   ├── config.py
│   ├── data/tiles.py                 # tiling, spatial block splits, buffer
│   ├── data/domars.py
│   ├── data/ai4mars.py               # optional: mask merging + augmentations
│   ├── terrain/slope.py
│   ├── terrain/hazard_labels.py      # slope → classes + ignore mask
│   ├── terrain/roughness.py          # stretch
│   ├── terrain/costmap.py
│   ├── terrain/route.py
│   ├── terrain/compare_routes.py     # overlap, extra cost, hazard crossings
│   ├── models/hazard_unet.py
│   ├── models/landform.py            # classifier + sliding-window mapper
│   ├── models/ground_unet.py         # optional
│   ├── eval/metrics.py               # IoU, confusion, aspect stratification
│   ├── eval/baselines.py             # always-safe, image gradient
│   └── viz/overlays.py
├── train/
│   ├── train_hazard.py               # CLI, AMP, --subset, --epochs, best ckpt by val mIoU
│   ├── train_landform.py
│   └── train_ground.py               # optional
├── app/app.py
├── notebooks/                        # exploration only
└── tests/   # hazard labels, splits/leakage, costmap, route, route comparison, AI4Mars mask merge
```

Raster cropping/resampling uses rasterio (no GDAL command-line tools are installed).

---

## 5. Timeline

### Pre-event prep (before Nov 14: allowed work only)
- [ ] **Confirm the rules on pre-event code with the local lead** (and whether an empty repo skeleton counts)
- [ ] Read the official challenge statement + resources and the **Nov 13 participant guides**; update the §0 wording
- [ ] Environment: Python 3.13 venv with CUDA torch, install `requirements.txt`; HF, Colab and Kaggle accounts with GPU access
- [ ] Downloads:
  - [ ] HiRISE TRN DTM (1.8 GB)
  - [ ] HiRISE TRN orthomosaic (7.4 GB; URL in §2): check it covers the DTM (`rio info`)
  - [ ] CTX of Jezero
  - [ ] DoMars16k `data.zip` + `models.zip`
  - [ ] (optional) AI4Mars; one small second-site DTM + ortho pair from USGS STAC
- [ ] Crop the DTM and CTX to the overlap region / AOI, and resample the ortho to 1 m, with rasterio → `data/processed/`
  (no GDAL CLI installed); then delete the raw 25 cm ortho if disk is tight
- [ ] Check with `rio info` whether the DTM mosaic has a footprint/mask layer for blend zones and seams
- [ ] Skim the DoMars16k paper (methods + Jezero figure); optionally AI4Mars

### Saturday Nov 14
Priority (= cut order, see below): **(a)** DTM slope + A* → **(b)** landform → **(c)** hazard U-Net → **(d)** DTM-free
comparison → **(e)** app polish + deploy → **(f)** AI4Mars tab. U-Net training starts early because it runs on the GPU
in parallel; that's scheduling, not a change of priority.

| Time | Milestone | Done when |
|---|---|---|
| 0–1.5 h | (a) Slope + hazard labels + cost + A* on the DTM | Route avoids steep slopes on a hillshade; class histogram checked |
| 1.5–3 h | (c) Tiles + spatial splits → **start hazard U-Net training on GPU** | Leakage test passes; loss falling |
| 3–6 h | (b) Landform map → combined cost | Landform raster plausible next to CTX; route changes with landform |
| 6–8 h | (c) Evaluate U-Net: mIoU, baselines, aspect table | Beats both baselines on the test region (or we say it doesn't) |
| 8–11 h | (d) DTM-free mode: routes T / O / P + comparison stats | Three routes on the map + stats table |
| 11–13 h | (e) Gradio app v1 running locally | Click-to-route in both modes |
| Night | Longer U-Net run; (f) AI4Mars training on a second GPU if spare | Better checkpoint by morning |

### Sunday Nov 15
| Time | Milestone |
|---|---|
| Morning | (e) **Deploy to HF Spaces**; final test-region eval with the best checkpoint; second-site test (stretch) |
| Midday | (e) Polish overlays/legend, How-it-works tab; (f) AI4Mars tab only if everything above is done |
| Afternoon | README (incl. limitations), project page text, slides, **30-second demo video** |
| Evening | Buffer for bugs. **Submit hours before 11:59 PM** |

### Cut list (cut from the top first)
1. (f) AI4Mars ground-view tab
2. (e) Polish (keep a minimal deployed app)
3. (d) Comparison extras: oracle route O, fixed-pair summary table (keep routes T vs. P + basic stats)
4. (c) Stretch items: second site, roughness, 0.5 m input, baseline-length ablation
5. (c) The hazard U-Net itself (fall back to "image-gradient baseline vs. DTM")
6. (b) Landform layer

**Never cut:** (a) DTM slope + A* route, a minimal app and the deployed link.

---

## 6. Judging angle / story
1. **Problem:** Astronauts on a Marswalk can't afford to get stuck or fall. Spirit got trapped in sand; steep, loose slopes are a real EVA hazard.
2. **Gap:** Slope needs a DTM, and HiRISE DTMs cover only a small part of Mars. Images cover far more.
3. **Solution:** A U-Net that predicts slope hazard from one image, plus a landform layer → hazard-aware routes even where no DTM exists.
4. **Proof:** mIoU on a geographically held-out region vs. two baselines, and DTM-free vs. DTM routes side by side
   (overlap, extra cost, hazards crossed).
5. **Honesty slide:** single-image slope depends on illumination (aspect-stratified results), 3 m baseline and
   thresholds are assumptions, one site (+ second-site result if done), cost weights are assumptions.
6. **Future:** more HiRISE DTM sites for training, sun-angle as an input, shape-from-shading priors, boulder detection, HUD for suits.

---

## 7. Working with Claude Code in VS Code
- Keep `CLAUDE.md` at the repo root. Claude Code reads it automatically every session.
- **One milestone per prompt**, e.g. *"Implement src/martianmap/terrain/hazard_labels.py per PLAN.md §3C, with tests."*
- Use plan mode for bigger pieces (tiling + splits, sliding-window mapper, route comparison, app layout).
- Tests for the pure-logic parts: hazard labels, splits/leakage, cost map, route, route comparison, mask merging.
- **Commit after every working milestone.**
- Training on Colab/Kaggle: `!python train/train_hazard.py --subset 50 --epochs 1` to smoke-test, then the full run.
- When something breaks, paste the full traceback and the command you ran.
- No multi-GB downloads mid-session; use the prepped `data/processed/`.

---

## 8. Key references
- USGS, *Mars 2020 Terrain Relative Navigation HiRISE DTM Mosaic*, doi:10.5066/P9REJ9JN (+ companion orthomosaic)
- Wilhelm et al., *DoMars16k*, Remote Sensing 12(23), 2020, doi:10.3390/rs12233981 · Zenodo 4291940 · github.com/thowilh/geomars
- Horn, *Hill shading and the reflectance map*, Proc. IEEE 69(1), 1981 (slope operator)
- USGS Astrogeology STAC catalog: stac.astrogeology.usgs.gov
- Swan et al., *AI4MARS*, CVPR Workshops 2021 (optional tab)
- Purohit et al., *Mars-Bench*, NeurIPS 2025 Datasets & Benchmarks (stretch)
