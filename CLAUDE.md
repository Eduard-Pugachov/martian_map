# CLAUDE.md — Martian Map (NASA Space Apps 2026)

## What this project is
Hackathon entry for "Interplanetary Survival Guide: Martian Map". A hazard-aware Marswalk route planner for Jezero
Crater whose main ML piece predicts slope hazard from a single HiRISE image, so routes can be planned where no DTM exists.
Full plan: **PLAN.md** (read the relevant section before coding).

- **A. Hazard U-Net (main):** HiRISE orthoimage (1 m) → per-pixel safe / caution / hazard. Labels come from DTM slope.
- **B. Landform classifier:** 15-class CTX patch classifier (DoMars16k) → sliding-window landform map.
- **C/D. Terrain + routing:** cost = f(slope hazard, landform) → least-cost route (skimage `route_through_array`).
  - DTM mode: cost from true DTM slope.
  - **DTM-free mode:** cost from the U-Net's predicted hazard. Compare routes T (true), O (oracle classes), P (predicted).
- **E/F. App:** Gradio Blocks on Hugging Face Spaces: Mission Map (with DTM-free toggle), How it works, optional Ground view.
- **G. Ground-view U-Net (optional):** AI4Mars soil / bedrock / sand / big rock, a small demo tab only.

## Status
Coding has started (Eddie's call, Nov 13). Local env: `.venv` (Python 3.13, CUDA torch) in the repo root, gitignored.
Raw data lives in `data/raw/jezero/` and `data/raw/domars16k/`. C: is nearly full (~27 GB free before downloads):
check free space before writing large files. The raw 25 cm ortho was deleted (Nov 14) after `ortho_1m.tif` was
built, so `precompute_layers.py --layers ortho` needs a re-download first (URL in PLAN.md §2).

## Priorities (also the cut order, from the bottom)
(a) DTM slope + A* route → (b) DoMars16k landform layer → (c) hazard U-Net → (d) DTM-free comparison →
(e) app polish + deployment → (f) AI4Mars tab.
**Never cut:** (a), a minimal app and the deployed link. Starting U-Net training early on the GPU is fine (parallel).

## Hard deadline
Submission due Sun Nov 15, 2026, 11:59 PM local. A working demo matters more than elegance. Prefer simple, proven libraries.

## Conventions
- Python 3.13. Package code lives in `src/martianmap/`. Notebooks are for exploration only.
- All tunable numbers (AOI, split blocks, buffer, slope baseline, hazard thresholds, cost weights, paths, class maps)
  go in `config.yaml`, never hard-coded.
- Type hints + short docstrings. Keep functions small and pure where possible.
- Data in `data/raw` and `data/processed` is gitignored. **Never download multi-GB files without asking.**
- Use `pathlib`, `logging` (no bare prints in src/), `tqdm` for loops over data.
- Tests in `tests/` with pytest; required for hazard labels, spatial splits (no leakage), cost map, route,
  route comparison, and AI4Mars mask merging.

## Data facts
- HiRISE TRN DTM: 1 m/px GeoTIFF, equirectangular (Mars sphere). It's a mosaic blended with MOLA ("blend40"):
  mask no-data, blend zones and seams between source DTMs as 255 (ignore).
- HiRISE TRN orthomosaic: `JEZ_hirise_soc_006_orthoMosaic_25cm_Eqc_latTs0_lon0_first.tif` (7.4 GB, same `soc_006`
  production as the DTM). Resample (area average) onto the DTM's 1 m grid with rasterio, reading in windows (never
  load the full 25 cm raster into memory); grids may not align exactly, but no feature matching is needed.
- **Grids line up exactly** (checked Nov 13): DTM 21400×21488 px float32 at 1 m, nodata ≈ -3.4028e38; ortho
  85600×85952 px uint8 at 0.25 m, nodata 0; identical bounds (x 4577366–4598766, y 1085130–1106618 m).
  So ortho → 1 m is an exact 4×4 block average (ignore nodata 0), no reprojection. Neither file has overviews.
- CTX: `JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0.tif` (36 MB, 6 m/px, 5322×5040, covers more than the DTM).
- DoMars16k `models.zip` holds PyTorch `.pth` weights: resnet18, resnet50, densenet121, densenet161, vgg16, alexnet.
- No GDAL command-line tools are installed; use rasterio (Python or its `rio` CLI) for raster work.
- Slope: Gaussian smooth (σ = 1 px) + Horn 3×3 → effective baseline ≈ 3 m (DTM resolves ~3–5 m features; 1 m is noise).
- Hazard classes: 0 = safe (< 10°), 1 = caution (10–25°), 2 = hazard (≥ 25°), 255 = ignore. Hazard is rare → class weights.
  Measured on the full DTM (Nov 14): 66% of pixels valid; of valid: safe 77.4%, caution 21.2%, hazard 1.40%
  (≥ 20°: 4.3%, ≥ 30°: 0.4%). Ripple fields on the western floor show up as speckled caution (real slopes, not a seam).
- Precomputed: `data/processed/slope_deg.tif` (float32, NaN = ignore) and `hazard_labels.tif` (uint8, 255 = ignore),
  same grid as the DTM. Made by `scripts/precompute_layers.py --layers slope hazard` (~3 min, strip-wise, low RAM).
- **Splits are spatial blocks, never random tiles:** 1 km blocks, contiguous test and val regions, 300 m buffer dropped
  between splits. The routing demo AOI (~4×4 km) lies inside the test region.
  Built (Nov 14) by `scripts/make_hazard_tiles.py` → `data/processed/tiles.csv` (index only; read windows from
  `ortho_1m.tif` + `hazard_labels.tif`): 512 px tiles, train 869 / val 73 / test 81 (85 / 7 / 8% of px),
  hazard 1.42 / 1.77 / 1.40%. Test blocks rows 8–13 × cols 8–13 hold the whole demo route; val = northern
  scarps (rows 0–4 × cols 11–16). Train tiles stay ≥ 300 m from the test/val *blocks*, so the whole test
  block area (not just test tiles) is unseen ground.
- Rotations/flips are valid augmentations for the hazard U-Net (slope magnitude doesn't change under rotation).
- **Illumination caveat:** single-image slope depends on sun direction, mixed sun angles across the mosaic, and albedo.
  Always report test IoU stratified by DTM aspect, and state the limitation in the README/app.
- DoMars16k: 200×200 px CTX patches (~1.2 km), 15 classes (aec, ael, cli, rid, fsf, sfe, fsg, fse, fss, cra, sfx, mix, rou, smo, tex).
- AI4Mars (optional): labels 0 = soil, 1 = bedrock, 2 = sand, 3 = big rock, 255 = null; merge rover (MXY) and range (RNG)
  masks into 255. NAVCAM is grayscale → repeat to 3 channels. No vertical flips.

## Models
- Hazard U-Net: `smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=3)`, tiles 512 (or 256),
  CE with class weights + Dice, `ignore_index=255`. Baselines: always-safe, image-gradient (Sobel, thresholds fit on train).
  Run 1 (Nov 14, `train/train_hazard.py --epochs 40`, ~9 min on the RTX 4080 Laptop): best val mIoU 0.542
  (safe 0.80 / caution 0.40 / hazard 0.43) at epoch 40 → `checkpoints/hazard_unet_best.pt` (+ `.csv` log).
  Predictions are blobby: big scarps found, fine ripples mostly missed. The model calls the ortho's nodata edge
  a scarp → always mask ortho == 0 (and a small margin) when making predicted-hazard maps.
  Eval (Nov 14, `scripts/eval_hazard.py`, whole 6×6 km test region, unseen): U-Net mIoU **0.547**
  (safe 0.86 / caution 0.42 / hazard 0.36, hazard recall 71%) vs gradient baseline 0.288 and always-safe 0.283.
  Gradient baseline is genuinely weak (oracle-tuned on test: 0.289; fitter verified against brute force).
  Illumination: **west-facing slopes are much worse** (test hazard IoU 0.14 / recall 48% vs 0.35–0.42 / 60–78%
  elsewhere; val shows the same). Results: `data/processed/eval/hazard_{test,val}.json`, `previews/eval_*.png`.
- Landform (Nov 14): pretrained DoMars16k `resnet50.pth` (keys prefixed `net.`; preprocessing = resize 224 +
  ImageNet norm; classes in alphabetical ImageFolder order) → 93.2% on the DoMars16k test split.
  `precompute_layers.py --layers landform` → `landform.npz`: 75×75 cells of 300 m over the DTM, each averaging all
  overlapping 1.2 km windows; cells with max prob < 0.6 are UNKNOWN (multiplier 1.0), which kills low-confidence
  'cliff' misfires on the flat floor. 61% of cells confident: tex 48%, mix 28%, fss 11%, sfx 10%. 'fss' (mass
  wasting) on the crater floor looks like domain shift. The landform term does not change the main demo route;
  the second pair `route.landform_demo` (3.6 km, SW floor) shows it: slope-only crosses 17% dunes, fused 0%.
  Class codes per the Zenodo record (doi 10.5281/zenodo.4291940), not the names' initials: aec = aeolian straight,
  ael = aeolian curved, fsf = channel, sfe = mounds, fsg = gullies, fse = slope streaks, fss = mass wasting.
  UNKNOWN cells cost ×1.0, cheaper than every class but smooth, so routes can drift into low-confidence cells.
- DTM-free mode (Nov 14): `precompute_layers.py --layers pred` → `pred_hazard.npz` (U-Net classes over the test
  region only, 5 m cell = max class, 1200×1200). `scripts/compare_routes.py --preview` → routes T (true slope),
  O (oracle classes), P (predicted), L (landform only, no hazard info, baseline), scored on the unbuffered true
  cost → `data/processed/eval/route_compare.json`, `previews/route_compare.png`.
  20 fixed pairs (2–6 km): P crosses a true hazard on 20% of pairs (L: 80%, O: 0%); P median extra cost 16% when
  passable. Demo pair: P crosses 9 hazard cells (one ~60 m scarp face the U-Net called caution), L crosses 29.
  5 m path overlap is low even for O (median 15%): the floor has many near-equal corridors.
- Ground U-Net (optional): `smp.Unet("resnet34", encoder_weights="imagenet", classes=4)`.
- Training scripts are CLI tools (argparse) that run on Colab/Kaggle GPU: AMP, `--subset`, `--epochs`, best ckpt by val mIoU/acc.
- The app runs on CPU. Predicted hazard, slope, landform and cost rasters are precomputed; the app only runs A*.

## How to work with me (Eddie)
- One milestone at a time; propose a short plan before large changes.
- After each milestone: tell me exactly what command to run to verify it, then I commit.
- If an assumption is needed (e.g. thresholds, cost weights, split regions), put it in config.yaml and flag it in your reply.
