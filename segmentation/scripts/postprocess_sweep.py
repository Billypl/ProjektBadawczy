"""Tune the small-blob filter (usvseg.postprocess) on existing prediction PNGs.

No model inference: reads runs/<src>/preds/<src>/*.png, applies each BlobFilter of
the grid, writes runs/<src>_<tag>/preds/<src>_<tag>/ and runs the official evaluator
through usvseg.eval.run_lars_eval (rows go to results/dev_selection.csv by default).

    python scripts/postprocess_sweep.py --src m1_segformer_b2_e80_dev --model segformer_b2 \
        --areas 50 100 200 500 1000 --dists 0 10 25
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from usvseg.data.taxonomy import color_to_labels, labels_to_color
from usvseg.postprocess import BlobFilter, remove_small_blobs

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="experiment whose predictions are filtered")
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-data", default="LaRS train (-dev)")
    ap.add_argument("--split", default="dev", choices=["dev", "val"])
    ap.add_argument("--areas", type=int, nargs="+", default=[50, 100, 200, 500, 1000])
    ap.add_argument("--dists", type=int, nargs="+", default=[0, 10, 25])
    ap.add_argument("--results-csv", type=Path, default=REPO / "results" / "dev_selection.csv")
    args = ap.parse_args()

    src_dir = REPO / "runs" / args.src / "preds" / args.src
    pngs = sorted(src_dir.glob("*.png"))
    if not pngs:
        sys.exit(f"no predictions in {src_dir}")
    preds = {p.stem: color_to_labels(np.asarray(Image.open(p).convert("RGB"))) for p in tqdm(pngs, desc="load")}

    for area in args.areas:
        for dist in args.dists:
            f = BlobFilter(max_area=area, shore_dist=dist)
            exp = f"{args.src}_{f.tag()}"
            exp_dir = REPO / "runs" / exp
            out_dir = exp_dir / "preds" / exp
            out_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "config.yaml").write_text(yaml.safe_dump(
                {"src": args.src, **vars(f), "split": args.split}, sort_keys=False))
            changed = 0
            for name, labels in tqdm(preds.items(), desc=exp):
                out = remove_small_blobs(labels, f)
                changed += int((out != labels).sum())
                Image.fromarray(labels_to_color(out)).save(out_dir / f"{name}.png")
            print(f"{exp}: {changed / len(preds):.0f} px/frame relabelled", flush=True)
            subprocess.run([sys.executable, "-m", "usvseg.eval.run_lars_eval", "--exp", exp,
                            "--model", args.model, "--train-data", args.train_data, "--split", args.split,
                            "--results-csv", str(args.results_csv),
                            "--notes", f"small-blob filter max_area {area} shore_dist {dist} on {args.src}"],
                           cwd=REPO, check=True)


if __name__ == "__main__":
    main()
