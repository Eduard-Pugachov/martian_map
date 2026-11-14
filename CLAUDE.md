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
check free space before writing large files, and prefer deleting the raw 25 cm ortho once the 1 m version exists.

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
- CTX: `JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0.tif` (36 MB, 6 m/px).
- No GDAL command-line tools are installed; use rasterio (Python or its `rio` CLI) for raster work.
- Slope: Gaussian smooth (σ = 1 px) + Horn 3×3 → effective baseline ≈ 3 m (DTM resolves ~3–5 m features; 1 m is noise).
- Hazard classes: 0 = safe (< 10°), 1 = caution (10–25°), 2 = hazard (> 25°), 255 = ignore. Hazard is rare → class weights.
- **Splits are spatial blocks, never random tiles:** 1 km blocks, contiguous test and val regions, 300 m buffer dropped
  between splits. The routing demo AOI (~4×4 km) lies inside the test region.
- Rotations/flips are valid augmentations for the hazard U-Net (slope magnitude doesn't change under rotation).
- **Illumination caveat:** single-image slope depends on sun direction, mixed sun angles across the mosaic, and albedo.
  Always report test IoU stratified by DTM aspect, and state the limitation in the README/app.
- DoMars16k: 200×200 px CTX patches (~1.2 km), 15 classes (aec, ael, cli, rid, fsf, sfe, fsg, fse, fss, cra, sfx, mix, rou, smo, tex).
- AI4Mars (optional): labels 0 = soil, 1 = bedrock, 2 = sand, 3 = big rock, 255 = null; merge rover (MXY) and range (RNG)
  masks into 255. NAVCAM is grayscale → repeat to 3 channels. No vertical flips.

## Models
- Hazard U-Net: `smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=3)`, tiles 512 (or 256),
  CE with class weights + Dice, `ignore_index=255`. Baselines: always-safe, image-gradient (Sobel, thresholds fit on train).
- Landform: try DoMars16k pretrained `models.zip` first, else `timm` resnet50/convnext_tiny fine-tune.
- Ground U-Net (optional): `smp.Unet("resnet34", encoder_weights="imagenet", classes=4)`.
- Training scripts are CLI tools (argparse) that run on Colab/Kaggle GPU: AMP, `--subset`, `--epochs`, best ckpt by val mIoU/acc.
- The app runs on CPU. Predicted hazard, slope, landform and cost rasters are precomputed; the app only runs A*.

## How to work with me (Eddie)
- One milestone at a time; propose a short plan before large changes.
- After each milestone: tell me exactly what command to run to verify it, then I commit.
- If an assumption is needed (e.g. thresholds, cost weights, split regions), put it in config.yaml and flag it in your reply.
