"""M2: YOLO-seg weights on LaRS -> semantic map -> evaluator-format PNGs.

Writes both unassigned-pixel variants in one pass (instances are predicted once):
    runs/<exp>_unobst/preds/<exp>_unobst/   unassigned -> OBSTACLE (cautious)
    runs/<exp>_unwater/preds/<exp>_unwater/ unassigned -> WATER

    python scripts/predict_yolo.py --exp m2_yolo26s_val --weights runs/m2_yolo26s_seg/weights/best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from common.interfaces import OBSTACLE, WATER
from usvseg.data.lars import DEFAULT_ROOT, Lars
from usvseg.eval.export_lars import save_prediction
from usvseg.models.yolo_seg import YoloSegmenter

REPO = Path(__file__).resolve().parents[1]
VARIANTS = {"unobst": OBSTACLE, "unwater": WATER}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help="smoke mode: only first N images")
    args = ap.parse_args()

    out_dirs = {}
    for suffix in VARIANTS:
        exp = f"{args.exp}_{suffix}"
        exp_dir = REPO / "runs" / exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "config.yaml").write_text(yaml.safe_dump(
            {**{k: str(v) for k, v in vars(args).items()}, "unassigned": suffix}, sort_keys=False))
        out_dirs[suffix] = exp_dir / "preds" / exp

    ds = Lars(args.split, args.root)
    samples = ds.samples[: args.limit] if args.limit else ds.samples
    seg = YoloSegmenter(args.weights, imgsz=args.imgsz, conf=args.conf)
    unassigned_pct = []
    for s in tqdm(samples, desc="YOLO"):
        img = s.image()
        hw = img.shape[:2]
        instances = seg.predict_instances(img)
        for suffix, unassigned in VARIANTS.items():
            seg.unassigned = unassigned
            save_prediction(seg.to_semantic(instances, hw), out_dirs[suffix], s.name, hw)
        covered = np.any([i.mask for i in instances], axis=0) if instances else np.zeros(hw, bool)
        unassigned_pct.append(100 * (1 - covered.mean()))
    print(f"{len(samples)} images, unassigned pixels: mean {np.mean(unassigned_pct):.2f}%, "
          f"max {np.max(unassigned_pct):.2f}%")


if __name__ == "__main__":
    main()
