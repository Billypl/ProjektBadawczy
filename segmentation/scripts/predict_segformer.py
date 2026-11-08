"""M1: SegFormer checkpoint on LaRS -> evaluator-format PNGs.

    python scripts/predict_segformer.py --exp m1_segformer_b2_val \
        --checkpoint runs/m1_segformer_b2/checkpoint --split val
"""

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

from usvseg.data.lars import DEFAULT_ROOT, Lars
from usvseg.eval.export_lars import save_prediction
from usvseg.models.segformer import SegformerSegmenter

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--scale", default="2048x1024", help="keep-ratio inference scale, long x short side")
    ap.add_argument("--limit", type=int, default=0, help="smoke mode: only first N images")
    args = ap.parse_args()

    scale = tuple(int(v) for v in args.scale.split("x"))
    exp_dir = REPO / "runs" / args.exp
    out_dir = exp_dir / "preds" / args.exp
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(
        {k: str(v) for k, v in vars(args).items()}, sort_keys=False))

    ds = Lars(args.split, args.root)
    samples = ds.samples[: args.limit] if args.limit else ds.samples
    seg = SegformerSegmenter(args.checkpoint, scale=scale)
    for s in tqdm(samples, desc="SegFormer"):
        img = s.image()
        labels, _ = seg.predict(img)
        save_prediction(labels, out_dir, s.name, img.shape[:2])
    print(f"{len(samples)} predictions -> {out_dir}")


if __name__ == "__main__":
    main()
