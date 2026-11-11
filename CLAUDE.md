# CLAUDE.md — Martian Map (NASA Space Apps 2026)

## What this project is
Hackathon entry for "Interplanetary Survival Guide: Martian Map". An ML-driven, hazard-aware
Marswalk route planner for Jezero Crater. Full plan: **PLAN.md** (read the relevant section before coding).

- **A. Ground U-Net:** segments rover images into soil / bedrock / sand / big rock (AI4Mars).
- **B. Landform classifier:** 15-class CTX patch classifier (DoMars16k) → sliding-window landform map.
- **C/D. Terrain:** slope from the HiRISE DTM + landform → cost map → least-cost route (skimage `route_through_array`).
- **E. App:** Gradio Blocks, deployed to Hugging Face Spaces.

## Hard deadline
Submission due Sun Nov 15, 2026, 11:59 PM local. A working demo matters more than elegance. Prefer simple, proven libraries.

## Conventions
- Python 3.11. Package code lives in `src/martianmap/`. Notebooks are for exploration only, not logic.
- All tunable numbers (AOI bounds, cost weights, slope limits, paths, class maps) go in `config.yaml`, never hard-coded.
- Type hints + short docstrings. Keep functions small and pure where possible (easy to test).
- Data in `data/raw` and `data/processed` is gitignored. **Never download multi-GB files without asking.**
- Use `pathlib`, `logging` (no bare prints in src/), `tqdm` for loops over data.
- Tests in `tests/` with pytest; required for costmap, route, and mask-merging logic.

## Data facts
- AI4Mars labels: 0=soil, 1=bedrock, 2=sand, 3=big rock, 255=null. Merge rover mask (MXY) and
  30 m range mask (RNG) into 255. Loss uses `ignore_index=255`. Big rock is rare → class weights.
- NAVCAM is grayscale → repeat to 3 channels for ImageNet encoders.
- DoMars16k: 200×200 px CTX patches, 15 classes (aec, ael, cli, rid, fsf, sfe, fsg, fse, fss, cra, sfx, mix, rou, smo, tex).
- DTM: 1 m/px GeoTIFF (rasterio). Always work on the cropped AOI in `data/processed/`.

## Models
- U-Net: `segmentation_models_pytorch.Unet("resnet34", encoder_weights="imagenet", classes=4)`.
- Landform: try DoMars16k pretrained `models.zip` first, else `timm` resnet50/convnext_tiny fine-tune.
- Training scripts are CLI tools (argparse) that must run on Colab/Kaggle GPU: AMP, `--subset`, `--epochs`, save best ckpt by val mIoU/acc.
- The app must run inference on CPU. Heavy layers are precomputed by `scripts/precompute_layers.py`.

## How to work with me (Eddie)
- One milestone at a time; propose a short plan before large changes.
- After each milestone: tell me exactly what command to run to verify it, then I commit.
- If an assumption is needed (e.g. cost weights), put it in config.yaml and flag it in your reply.
