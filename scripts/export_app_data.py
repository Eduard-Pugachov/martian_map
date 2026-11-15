"""Bundle the precomputed layers the app needs (test region, 5 m cells) into one small .npz.

The app runs on CPU and only does A*, so it gets arrays, not models or the 1 m rasters.
Usage: python scripts/export_app_data.py
Needs: precompute_layers.py --layers cost landform pred (and ortho_1m.tif)
"""

import argparse

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from skimage.measure import block_reduce

from martianmap.config import load_config, repo_path
from martianmap.models.landform import CLASSES as LANDFORM_CLASSES
from martianmap.models.landform import sample_to_grid


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    cfg = load_config()
    p = cfg["paths"]
    g = np.load(repo_path(p["route_grid"]))
    pr = np.load(repo_path(cfg["dtm_free"]["pred_maps"]["test"]))
    pred, r0, c0 = pr["cls"], int(pr["row0"]), int(pr["col0"])
    h, w = pred.shape
    cell = float(g["cell_m"])
    f = round(cell / cfg["grid"]["dtm_pixel_size_m"])
    transform = Affine(*pr["transform"])
    lf_map = np.load(repo_path(cfg["domars"]["map"]))
    landform = sample_to_grid(lf_map["cls"], Affine(*lf_map["transform"]), transform, (h, w))
    with rasterio.open(repo_path(p["ortho_1m"])) as o:
        img = o.read(1, window=Window(c0 * f, r0 * f, w * f, h * f)).astype(np.float32)
    ortho = block_reduce(img, (f, f), np.mean).astype(np.uint8)  # basemap only; nodata blocks stay dark
    out = repo_path(cfg["app"]["bundle"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        slope=g["slope"][r0 : r0 + h, c0 : c0 + w],
        dem=g["dem"][r0 : r0 + h, c0 : c0 + w],
        ortho=ortho,
        landform=landform,
        landform_classes=np.array(LANDFORM_CLASSES),
        pred=pred,
        p_hazard=pr["probs"][2],
        transform=np.array(transform)[:6],
        cell_m=cell,
    )
    print(f"wrote {out}  {h}x{w} cells at {cell:g} m, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
