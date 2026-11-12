"""Convert LaRS (train/dev/val) to an Ultralytics YOLO-seg dataset.

    python scripts/convert_lars_yolo.py [--out data/yolo/lars] [--epsilon 1.0] [--no-keyhole]

Writes images (symlinks) and labels, data.yaml, per-image loss stats
(runs/checks/yolo/stats.csv) and 10 overlays of the rasterised polygons.
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from usvseg.data.lars import DEFAULT_ROOT, Lars
from usvseg.data.to_yolo import (YOLO_NAMES, convert_image, from_yolo_lines, rasterize,
                                 semantic_from_class_masks, to_yolo_lines)
from usvseg.viz.overlay import overlay, side_by_side

REPO = Path(__file__).resolve().parents[1]
CHECKS = REPO / "runs" / "checks" / "yolo"


def convert_one(args, out: Path, split: str, segments: dict, keyhole: bool, epsilon: float):
    sample = args
    pan = sample.panoptic_ids()
    polys, st = convert_image(pan, segments[sample.name], sample.semantic(), keyhole=keyhole, epsilon=epsilon)
    (out / "labels" / split / f"{sample.name}.txt").write_text("\n".join(to_yolo_lines(polys, pan.shape)) + "\n")
    link = out / "images" / split / sample.image_path.name
    if not link.exists():
        os.symlink(sample.image_path.resolve(), link)
    return {"split": split, "image": sample.name, "n_polygons": st.n_polygons, "n_holes": st.n_holes,
            "n_tiny": st.n_tiny, "semantic_changed_pct": st.semantic_changed_pct,
            "n_things": len(st.thing_iou),
            "thing_iou_min": min(st.thing_iou) if st.thing_iou else np.nan,
            "thing_iou_mean": float(np.mean(st.thing_iou)) if st.thing_iou else np.nan}


def write_overlays(out: Path, ds: Lars, n: int = 10) -> None:
    CHECKS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for s in rng.choice(ds.samples, size=n, replace=False):
        img = s.image()
        hw = img.shape[:2]
        lines = (out / "labels" / "train" / f"{s.name}.txt").read_text().split("\n")
        rebuilt = semantic_from_class_masks(rasterize(from_yolo_lines([l for l in lines if l], hw), hw), hw)
        panel = side_by_side([overlay(img, s.semantic()), overlay(img, rebuilt)], ["GT semantic", "from YOLO polygons"])
        Image.fromarray(panel).save(CHECKS / f"{s.name}.jpg", quality=90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "yolo" / "lars")
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--no-keyhole", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = []
    for split in ("train", "dev", "val"):
        ds = Lars(split, args.root)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        segments = {a["file_name"].rsplit(".", 1)[0]: a["segments_info"]
                    for a in ds.panoptic_annotations["annotations"]}
        fn = partial(convert_one, out=args.out, split=split, segments=segments,
                     keyhole=not args.no_keyhole, epsilon=args.epsilon)
        with Pool(args.workers) as pool:
            rows += list(tqdm(pool.imap_unordered(fn, ds.samples, chunksize=8), total=len(ds), desc=split))

    (args.out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(args.out.resolve()), "train": "images/train", "val": "images/dev",
        "test": "images/val", "names": dict(enumerate(YOLO_NAMES))}, sort_keys=False))

    df = pd.DataFrame(rows).sort_values(["split", "image"])
    CHECKS.mkdir(parents=True, exist_ok=True)
    df.to_csv(CHECKS / "stats.csv", index=False)
    summary = df.groupby("split").agg(
        images=("image", "count"), polygons=("n_polygons", "sum"), holes=("n_holes", "sum"),
        tiny=("n_tiny", "sum"), changed_mean=("semantic_changed_pct", "mean"),
        changed_max=("semantic_changed_pct", "max"), things=("n_things", "sum"),
        thing_iou_min=("thing_iou_min", "min"), thing_iou_mean=("thing_iou_mean", "mean"))
    print(summary.round(3).to_markdown())
    write_overlays(args.out, Lars("train", args.root))
    print(f"dataset -> {args.out}, stats/overlays -> {CHECKS}")


if __name__ == "__main__":
    main()
