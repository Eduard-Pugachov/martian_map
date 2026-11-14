"""Train the image-only slope hazard U-Net on the spatial-split tiles.

Smoke test: python train/train_hazard.py --subset 16 --epochs 1 --device cpu --workers 0
Full run:   python train/train_hazard.py --epochs 30
"""

import argparse
import csv
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from martianmap.config import load_config, repo_path
from martianmap.data.hazard_dataset import HazardTiles
from martianmap.eval.metrics import confusion, iou_per_class
from martianmap.models.hazard_unet import build_model
from martianmap.terrain.hazard_labels import IGNORE

CLASSES = ("safe", "caution", "hazard")


def load_tiles(path) -> dict[str, list[dict]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}


def class_weights(rows: list[dict]) -> torch.Tensor:
    """1/sqrt(pixel frequency) over train tiles, scaled so the commonest class = 1."""
    freq = np.array([sum(float(r[c]) * float(r["valid"]) for r in rows) for c in CLASSES])
    w = 1 / np.sqrt(freq / freq.sum())
    return torch.tensor(w / w.min(), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, amp) -> torch.Tensor:
    model.eval()
    cm = torch.zeros(3, 3, dtype=torch.long, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            pred = model(x).argmax(1)
        cm += confusion(pred, y, 3, IGNORE)
    return cm.cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--subset", type=int, default=0, help="use only the first N train and val tiles (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/hazard_unet_best.pt")
    args = ap.parse_args()

    cfg = load_config()
    p, tile = cfg["paths"], cfg["tiles"]["size_px"]
    torch.manual_seed(0)
    tiles = load_tiles(repo_path(p["tiles_index"]))
    if args.subset:
        tiles = {s: v[: args.subset] for s, v in tiles.items()}
    ortho, labels = repo_path(p["ortho_1m"]), repo_path(p["hazard_labels"])
    loaders = {
        s: DataLoader(HazardTiles(ortho, labels, [(int(r["row"]), int(r["col"])) for r in tiles[s]], tile, train=s == "train"),
                      batch_size=args.batch_size, shuffle=s == "train", num_workers=args.workers,
                      persistent_workers=args.workers > 0, pin_memory=args.device == "cuda", drop_last=s == "train")
        for s in ("train", "val")
    }

    device = torch.device(args.device)
    amp = device.type == "cuda"  # bf16 autocast: no GradScaler needed
    model = build_model().to(device)
    weights = class_weights(tiles["train"]).to(device)
    ce = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE)
    dice = smp.losses.DiceLoss("multiclass", ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loaders["train"]))
    print(f"device {device} | train {len(tiles['train'])} / val {len(tiles['val'])} tiles | "
          f"class weights {[round(w, 2) for w in weights.tolist()]}")

    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".csv")
    best = -1.0
    with open(log_path, "w", newline="") as logf:
        log = csv.writer(logf)
        log.writerow(["epoch", "train_loss", "val_miou", *[f"val_iou_{c}" for c in CLASSES], "seconds"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0, losses = time.time(), []
            for x, y in loaders["train"]:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    logits = model(x)
                    loss = ce(logits.float(), y) + dice(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                losses.append(loss.item())
            iou = iou_per_class(evaluate(model, loaders["val"], device, amp))
            miou = float(np.nanmean(iou.numpy()))
            secs = time.time() - t0
            log.writerow([epoch, f"{np.mean(losses):.4f}", f"{miou:.4f}", *[f"{v:.4f}" for v in iou.tolist()], f"{secs:.0f}"])
            logf.flush()
            mark = ""
            if miou > best:
                best = miou
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_miou": miou, "val_iou": iou.tolist(),
                            "class_weights": weights.tolist(), "args": vars(args)}, out)
                mark = "  * saved"
            print(f"epoch {epoch:3d}  loss {np.mean(losses):.3f}  val mIoU {miou:.3f}  "
                  + "  ".join(f"{c} {v:.3f}" for c, v in zip(CLASSES, iou.tolist())) + f"  {secs:.0f}s{mark}")
    print(f"best val mIoU {best:.3f} -> {out}\nlog -> {log_path}")


if __name__ == "__main__":
    main()
