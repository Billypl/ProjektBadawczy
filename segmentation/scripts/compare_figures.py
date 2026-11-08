"""Side-by-side comparison figures: image | GT | one panel per method.

    python scripts/compare_figures.py --methods m1_segformer_b2_val m2_yolo26s_val_unobst \
        --labels SegFormer-B2 YOLO26s-seg --n 10

Picks frames deterministically: half the frames where the methods disagree the most
on detection (|FP+FN difference| from evaluator frames.csv), half random.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from usvseg.data.lars import Lars
from usvseg.data.taxonomy import color_to_labels
from usvseg.viz.overlay import overlay, side_by_side

REPO = Path(__file__).resolve().parents[1]


def load_pred(exp: str, name: str) -> np.ndarray:
    return color_to_labels(np.asarray(Image.open(REPO / "runs" / exp / "preds" / exp / f"{name}.png").convert("RGB")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "figures" / "comparison")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = [pd.read_csv(REPO / "runs" / m / "eval" / m / "frames.csv", index_col=0) for m in args.methods]
    errors = pd.concat([f["FP"] + f["FN"] for f in frames], axis=1)
    spread = (errors.max(axis=1) - errors.min(axis=1)).sort_values(ascending=False)
    picked = list(spread.index[: args.n // 2])
    rest = [n for n in spread.index if n not in picked]
    picked += list(np.random.default_rng(args.seed).choice(rest, size=args.n - len(picked), replace=False))

    ds = Lars(args.split)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in picked:
        s = ds.by_name(name)
        img = s.image()
        panels = [img, overlay(img, s.semantic())] + [overlay(img, load_pred(m, name)) for m in args.methods]
        titles = ["image", "GT"] + [f"{lbl} (FP {f.FP[name]}, FN {f.FN[name]})"
                                    for lbl, f in zip(args.labels, frames)]
        Image.fromarray(side_by_side(panels, titles, height=300)).save(args.out / f"{name}.jpg", quality=90)
    print(f"{len(picked)} figures -> {args.out}")


if __name__ == "__main__":
    main()
