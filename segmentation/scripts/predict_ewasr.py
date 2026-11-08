"""M0 baseline: eWaSR (no IMU, MaSTr1325 weights) on LaRS -> evaluator-format PNGs.

Run with the eWaSR venv:
    third_party/eWaSR/.venv/bin/python scripts/predict_ewasr.py --exp m0_ewasr_val --split val
Then evaluate with the main venv:
    .venv/bin/python -m usvseg.eval.run_lars_eval --exp m0_ewasr_val --model ewasr_resnet18 \
        --train-data MaSTr1325 --split val --input-res 512x384
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

from usvseg.data.lars import DEFAULT_ROOT, Lars  # noqa: E402
from usvseg.eval.export_lars import save_prediction  # noqa: E402
from usvseg.models.ewasr import DEFAULT_WEIGHTS, EwasrSegmenter  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--input-hw", default="384x512", help="HxW fed to the model, or 'native'")
    ap.add_argument("--limit", type=int, default=0, help="smoke mode: only first N images")
    args = ap.parse_args()

    input_hw = None if args.input_hw == "native" else tuple(int(v) for v in args.input_hw.split("x"))
    exp_dir = REPO / "runs" / args.exp
    out_dir = exp_dir / "preds" / args.exp
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(
        {k: str(v) for k, v in vars(args).items()}, sort_keys=False))

    ds = Lars(args.split, args.root)
    samples = ds.samples[: args.limit] if args.limit else ds.samples
    seg = EwasrSegmenter(args.weights, input_hw=input_hw)
    for s in tqdm(samples, desc="eWaSR"):
        img = s.image()
        labels, _ = seg.predict(img)
        save_prediction(labels, out_dir, s.name, img.shape[:2])
    print(f"{len(samples)} predictions -> {out_dir}")


if __name__ == "__main__":
    main()
