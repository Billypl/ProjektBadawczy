"""Per-subset detection results from evaluator frames.csv + LaRS image_annotations.json.

Only aggregates the evaluator's per-frame TP/FP/FN counts, no metric is recomputed
from masks. F1 = 2TP / (2TP + FP + FN), as in the evaluator.

    python -m usvseg.eval.subsets --exp m0_ewasr_val [--split val]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from usvseg.data.lars import DEFAULT_ROOT, _find_file

REPO = Path(__file__).resolve().parents[2]
ATTRIBUTES = ("scene_type", "reflections", "lighting", "waves")


def breakdown(frames: pd.DataFrame, labels: dict[str, dict], key: str) -> pd.DataFrame:
    f = frames.assign(**{key: [labels[n].get(key) for n in frames.index]})
    g = f.groupby(key)[["TP", "FP", "FN"]].sum()
    g["Pr"] = 100 * g.TP / (g.TP + g.FP)
    g["Re"] = 100 * g.TP / (g.TP + g.FN)
    g["F1"] = 200 * g.TP / (2 * g.TP + g.FP + g.FN)
    # mean of per-frame WE_acc, not equal to the global mu from summary.json
    g["mu_frames"] = f.groupby(key)["WE_acc"].mean()
    g["n"] = f.groupby(key).size()
    return g.round(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()

    frames = pd.read_csv(REPO / "runs" / args.exp / "eval" / args.exp / "frames.csv", index_col=0)
    ann = json.loads(_find_file(args.root, "train" if args.split == "dev" else args.split,
                                "image_annotations.json").read_text())
    labels = {a["file_name"].rsplit(".", 1)[0]: a["labels"] for a in ann["annotations"]}
    for key in ATTRIBUTES:
        print(f"\n### {args.exp} by {key}\n")
        print(breakdown(frames, labels, key).to_markdown())


if __name__ == "__main__":
    main()
