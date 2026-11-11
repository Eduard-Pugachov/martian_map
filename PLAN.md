# Martian Map — ML Terrain Intelligence for Marswalk Planning
**NASA Space Apps 2026 · Challenge: "Interplanetary Survival Guide: Martian Map"**
Hackathon: Sat Nov 14 – Sun Nov 15, 2026 (submit by Nov 15, 11:59 PM local time; check your local event)

---

## 0. The pitch (one sentence)
Most teams on this challenge will put existing NASA layers (elevation, slope) on a map and draw a route.
**We make new layers with ML**: a landform classifier maps what the terrain *is* from orbit,
a U-Net labels what an astronaut would *see* at ground level (soil / bedrock / sand / big rock),
and both feed a hazard-aware route planner for a human Marswalk in **Jezero Crater**.

Why Jezero: Perseverance is there, the best public high-res elevation data covers it, and
DoMars16k's authors already ran a Jezero landform analysis, so we have something to check our results against.

> ⚠️ Before Saturday, read the **official challenge statement + resource list** on spaceappschallenge.org
> (full statements were released Oct 28). Adjust objectives and wording to match it, and use at least
> one dataset it lists. The challenge is about building a layered, integrated view of a Martian
> location or route, which this plan does.

---

## 1. System overview

```
             ORBITAL (map view)                          GROUND (astronaut view)
 ┌─────────────────────────────────────┐        ┌──────────────────────────────────┐
 │ CTX image of Jezero (~6 m/px)       │        │ Rover navcam images              │
 │   → Landform classifier (DoMars16k) │        │   → U-Net terrain segmentation   │
 │   → sliding window → landform map   │        │     (AI4Mars: soil/bedrock/      │
 │                                     │        │      sand/big rock)              │
 │ HiRISE DTM (1 m/px)                 │        │   → % sand / big rock = local    │
 │   → slope map                       │        │     hazard score + overlay       │
 └──────────────┬──────────────────────┘        └───────────────┬──────────────────┘
                │                                               │
                ▼                                               ▼
        Cost map = f(slope, landform class)            "What you'd see here" panel
                │
                ▼
     A* least-cost route (start → goal)  →  distance, elevation profile, est. walk time
                │
                ▼
            Gradio app (deployed on Hugging Face Spaces)
```

**Two ML components, on purpose:**
| Component | Task | Why it's here |
|---|---|---|
| **A. Ground terrain U-Net** | Pixel segmentation, 4 classes | Reuses your colorizer stack (PyTorch U-Net + Gradio). This is the "U-Net" part of the pitch. |
| **B. Orbital landform classifier** | 15-class patch classification → sliding-window map | The only good *orbital* labeled terrain dataset is patch-labeled, not pixel-labeled. This is what produces the map layer. |

(Honest note: no big public dataset has pixel-level sand/rock/bedrock labels *from orbit*.
AI4Mars is rover-level. So the orbital layer comes from a patch classifier, and the U-Net works on ground views.
Say this up front in the presentation; it makes the project look more credible, not less.)

---

## 2. Datasets

### Core (must have)
| # | Dataset | What it is | Use | Size / Access |
|---|---|---|---|---|
| 1 | **AI4Mars** (NASA JPL) | Curiosity NAVCAM/Mastcam images with crowd-sourced pixel labels: soil, bedrock, sand, big rock. ~16k labeled training images; 322-image expert "gold" test set | Train U-Net (A) | Hugging Face: `hassanjbara/AI4MARS` (parquet, easiest). Also on Kaggle: `yash92328/ai4mars-terrainaware-autonomous-driving-on-mars` |
| 2 | **DoMars16k** (TU Dortmund) | 16,150 CTX patches (200×200 px ≈ 1.2 km), 15 landform classes | Train/fine-tune landform classifier (B) | Zenodo record 4291940: `data.zip` (~128 MB) + **`models.zip` with pretrained networks** + code at github.com/thowilh/geomars |
| 3 | **Mars 2020 TRN HiRISE DTM mosaic** (USGS) | 1 m/px elevation of Jezero | Slope map | ~1.8 GB GeoTIFF: `https://planetarymaps.usgs.gov/mosaic/mars2020_trn/HiRISE/JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40.tif` |
| 4 | **CTX imagery of Jezero** | Input image for landform map | Run classifier (B) over it | USGS Jezero landing-site products (CTX orthomosaic / LVS map) at astrogeology.usgs.gov, or Murray Lab global CTX mosaic tiles |

**AI4Mars label encoding** (label PNGs): `0=soil, 1=bedrock, 2=sand, 3=big rock, 255=no label`.
Also ships rover masks (MXY) and 30 m range masks (RNG). **Set label to 255 where rover mask = 1
or range mask = 1**, and use `ignore_index=255` in the loss.

### Optional / stretch
| Dataset | Use |
|---|---|
| HiRISE orthomosaic of Jezero (25 cm/px, ~3.2 GB, USGS) | Pretty high-res basemap, and the input for boulder detection |
| **Mars-Bench** (NeurIPS 2025; mars-bench.github.io; HF + Zenodo) | `mb-boulder_seg` (HiRISE boulders, only 39 train images), `mb-crater_binary_seg`, `mb-s5mars`. Boulder seg = an extra orbital hazard layer |
| S5Mars (6,000 Curiosity images, 10 classes; on HF / FiftyOne zoo) | Finer-grained ground classes if AI4Mars works early |
| HiRISE landmark set v3.2 (Zenodo 4002935) | Craters/dunes classification at HiRISE scale |
| Perseverance raw images (mars.nasa.gov/mars2020/multimedia/raw-images) | Real Jezero ground images for the "what you'd see" panel. Expect domain shift (U-Net is trained on Curiosity) |
| NASA Mars Trek WMTS | Online basemap tiles if you go with a web map later |

### Download plan (do this BEFORE Saturday)
The big files (DTM 1.8 GB, ortho 3.2 GB, AI4Mars several GB) are the #1 thing that can kill a hackathon.
Download them Wed–Fri. **Crop the DTM and CTX to one area of interest (AOI)** (e.g. ~10×10 km around the
Perseverance landing site / delta front) and save small GeoTIFFs/NumPy arrays. The app should only ever touch the cropped files.

> Check your local event's rules on pre-work. Setting up your environment, downloading data and reading papers
> is normally fine; save the actual project code for the event weekend if the rules say so.

---

## 3. Component details

### A. Ground terrain U-Net (AI4Mars)
- **Library:** `segmentation_models_pytorch` (`smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=4)`).
  An ImageNet-pretrained encoder trains *much* faster than the scratch U-Net in your colorizer.
  Grayscale NAVCAM → repeat to 3 channels.
- **Input:** resize/crop to 512×512 (AI4Mars images are 1024×1024). Start at 256 if GPU time is tight.
- **Loss:** CrossEntropy(ignore_index=255) + Dice. **Class weights**: big rock is rare (~3% of pixels per JPL), so weight it up.
- **Augmentations (albumentations):** horizontal flip, random resized crop, brightness/contrast, slight blur. No vertical flips (the horizon matters).
- **Training:** AdamW lr 3e-4, cosine schedule, mixed precision, 10–20 epochs. Train on Colab/Kaggle T4 if you have no local GPU.
  **Use a subset (e.g. 5k images) first** to get a working checkpoint in under an hour, then do a full run while building the app.
- **Eval:** mIoU + per-class IoU on the expert test set (`test_min3` split). Save a confusion matrix image for the slides.
  Reference point: JPL's DeepLabv3 paper reports >96% pixel accuracy. You won't match it in a weekend, and you don't need to.
- **Output in app:** colored overlay + % of each class + "ground hazard score" = %sand·w_s + %bigrock·w_r.

### B. Orbital landform classifier (DoMars16k)
- **Fastest path:** use the **pretrained models in DoMars16k `models.zip`** first (check the format and architecture in the geomars repo).
  If loading them is a pain, fine-tune `timm` `resnet50` / `convnext_tiny` (ImageNet) on DoMars16k. That takes ~20–30 min on a T4 at 224 px.
- **Map generation:** slide a 200×200 window over the CTX AOI with stride 50–100 px → class probabilities per window → upsample to a class raster.
  Optional smoothing: majority filter (the paper uses Markov random fields; a 3×3 mode filter is fine for a hackathon).
- **Eval:** test-set accuracy + confusion matrix. Qualitative: compare your Jezero map to the figure in the DoMars16k paper.

### C. Slope & cost map
- Read the cropped DTM with `rasterio`; slope (degrees) = `arctan(hypot(dz/dx, dz/dy))` with `np.gradient` using pixel size (1 m).
- Resample the landform raster onto the DTM grid (or downsample both to a common ~5–10 m grid; this makes A* fast).
- **Cost per cell (tunable, document as assumptions):**
  - slope: `1 + (slope/10)^2`; **impassable if > 25°** (walking EVA assumption; flag it as tunable)
  - landform multiplier, e.g. smooth ×1.0, textured ×1.3, mixed ×1.5, rough ×2.5, crater/crater field ×2.0,
    aeolian curved/straight (dunes) ×4.0 (sand trapped Spirit & Opportunity, a good story beat), cliff = impassable, gullies/mass wasting ×3.0
- Put all weights in one `config.yaml` so they can be tuned live in the demo.

### D. Route planner
- `skimage.graph.route_through_array(cost, start, end, fully_connected=True, geometric=True)`, which is A*-equivalent least-cost path in one line.
- Compare **shortest straight-line path vs. safest path**: distance, total climb, max slope, % of path over dunes/rough terrain.
- Elevation profile along the path (matplotlib/plotly).
- Walk-time estimate: a Tobler-style hiking function scaled for Mars suits (flag it as a rough model).

### E. App (Gradio, since you already know it)
- **Tab 1 – Mission Map:** basemap image with toggleable overlays (slope / landform / cost). Click twice on the image (`gr.Image.select` gives pixel coords) for start & goal → draws both routes + stats table + elevation profile.
- **Tab 2 – Ground View:** pick a sample NAVCAM/Perseverance image or upload one → U-Net overlay + class percentages + hazard score.
- **Tab 3 – How it works / Model cards:** metrics, confusion matrices, dataset credits, assumptions.
- **Deploy to Hugging Face Spaces** (free, Gradio-native), so judges just open a link and don't have to install anything.
- Precompute heavy stuff (landform map, slope, cost raster) offline → ship as small `.npy`/PNG files. The app only runs A* + U-Net inference (CPU is fine).

---

## 4. Repo structure
```
martian-map/
├── CLAUDE.md                 # context for Claude Code (read first)
├── PLAN.md                   # this file
├── README.md                 # judge-facing: what, why, how to run, credits
├── requirements.txt
├── config.yaml               # AOI bounds, cost weights, slope limits, paths
├── data/                     # gitignored
│   ├── raw/                  # big downloads
│   └── processed/            # cropped AOI rasters, .npy, sample images
├── checkpoints/              # gitignored except final small weights
├── scripts/
│   ├── download_ai4mars.py
│   ├── download_domars.py
│   ├── download_jezero.py    # DTM + CTX, then crop to AOI
│   └── precompute_layers.py  # slope, landform map, cost map → data/processed
├── src/martianmap/
│   ├── data/ai4mars.py       # Dataset w/ mask merging + augmentations
│   ├── data/domars.py
│   ├── models/unet.py        # smp wrapper + load/save
│   ├── models/landform.py    # classifier + sliding-window mapper
│   ├── terrain/slope.py
│   ├── terrain/costmap.py
│   ├── terrain/route.py
│   └── viz/overlays.py       # colormaps, legend, route drawing
├── train/
│   ├── train_unet.py         # CLI args, AMP, logs mIoU, saves best ckpt
│   └── train_landform.py
├── app/app.py                # Gradio Blocks
├── notebooks/                # exploration only; logic lives in src/
└── tests/                    # costmap + route + mask-merging unit tests
```

**Core requirements:** `torch torchvision segmentation-models-pytorch timm albumentations rasterio numpy scipy scikit-image matplotlib plotly gradio datasets huggingface_hub pyyaml tqdm pytest`

---

## 5. Timeline

### Prep (Tue Nov 10 – Fri Nov 13)
- [ ] Register, form/confirm team, pick local or Universal Event
- [ ] Read the official challenge statement + resources; update §0 and README wording
- [ ] Set up Python 3.11 venv, `requirements.txt`, git repo, HF account, Colab/Kaggle GPU access
- [ ] Start ALL big downloads; pick the AOI; crop DTM + CTX
- [ ] Skim AI4Mars paper + DoMars16k paper (methods + Jezero figure)
- [ ] Open `models.zip` from DoMars16k: can you load a pretrained model? (decides Path B effort)

### Saturday Nov 14
| Time | Milestone | Done when |
|---|---|---|
| 0–2 h | Repo skeleton, config, data loaders + a notebook that visualizes AI4Mars samples with masks | Images and masks line up; rover/range masks merged into 255 |
| 2–4 h | **Kick off U-Net training on GPU** (subset first) | Loss going down; first checkpoint saved |
| 2–5 h (parallel) | Slope + cost map + A* on the DTM alone | A route that avoids steep slopes, plotted on a hillshade |
| 5–8 h | Landform classifier (pretrained or fine-tuned) + sliding-window Jezero map | Landform raster that looks plausible next to the CTX image |
| 8–10 h | Fuse landform into cost map; shortest vs. safest comparison | Two routes, visibly different, with a stats table |
| 10–13 h | Gradio app v1: Mission Map tab + Ground View tab running locally | Full click-to-route flow works |
| Night | Full U-Net training run | Better checkpoint by morning |

### Sunday Nov 15
| Time | Milestone |
|---|---|
| Morning | Eval: mIoU, confusion matrices, failure cases. Elevation profile + walk time. Polish overlays/legend |
| Midday | Deploy to HF Spaces; test on a clean browser/phone |
| Afternoon | README, project page text, slides, **30-second demo video** (screen-record the route flow) |
| Evening | Buffer for bugs. **Submit hours before 11:59 PM**, not minutes before |

**Cut list if behind (in order):** boulder layer → Perseverance images → walk time → landform smoothing → Tab 3.
**Never cut:** working route demo + U-Net ground view + deployed link.

---

## 6. Judging angle / story
1. **Problem:** Astronauts on a Marswalk can't afford to get stuck. Spirit got trapped in sand, and Curiosity's wheels were punctured by sharp rocks.
2. **Insight:** Elevation alone isn't enough; *what the ground is made of* matters. That's an ML problem.
3. **Solution:** Two models make two new terrain layers → hazard-aware route + ground-level preview.
4. **Proof:** Metrics on held-out NASA test sets + "shortest vs. safest" side-by-side.
5. **Honesty slide:** limitations (crowd labels, Curiosity → Perseverance domain shift, cost weights are assumptions).
6. **Future:** Mars foundation models (e.g. MOMO / Mars-Bench line of work), stereo-based rock sizing, HUD for suits.

---

## 7. Working with Claude Code in VS Code
- Keep `CLAUDE.md` at the repo root (included). Claude Code reads it automatically every session.
- **One milestone per prompt.** Example: *"Implement src/martianmap/data/ai4mars.py per PLAN.md §3A. Merge rover and range masks into 255. Add a test in tests/ that checks label values are in {0,1,2,3,255}. Then show me 4 samples in notebooks/01_ai4mars.ipynb."*
- Use **plan mode** (Shift+Tab) for bigger pieces (sliding-window mapper, app layout) so you approve the approach before it writes code.
- Ask for tests on the pure-logic parts (cost map, route, mask merging). They're cheap and catch silent bugs.
- **Commit after every working milestone** so it's easy to roll back if Claude Code goes off track.
- Training runs on Colab/Kaggle: have Claude Code make `train_unet.py` a CLI script with args, then run `!python train/train_unet.py --subset 5000 --epochs 5` in a notebook that clones your repo.
- When something breaks, paste the full traceback and the command you ran. Don't just describe it.
- Don't let Claude Code download multi-GB files in the middle of a session; use your prepped `data/processed/`.

---

## 8. Key references
- Swan et al., *AI4MARS: A Dataset for Terrain-Aware Autonomous Driving on Mars*, CVPR Workshops 2021
- Wilhelm et al., *DoMars16k*, Remote Sensing 12(23), 2020, doi:10.3390/rs12233981 · Zenodo 4291940 · github.com/thowilh/geomars
- USGS, *Mars 2020 Terrain Relative Navigation HiRISE DTM Mosaic*, doi:10.5066/P9REJ9JN
- Purohit et al., *Mars-Bench*, NeurIPS 2025 Datasets & Benchmarks · mars-bench.github.io
- Zhang et al., *S5Mars*, IEEE TGRS 2024
