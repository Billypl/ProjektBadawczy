"""Paired bootstrap over frames: is method B better than method A?

    python scripts/bootstrap_compare.py --a m1_segformer_b2_val --b m2_yolo26s_val_c010_unobst

Resamples frames with replacement from the evaluator's frames.csv (both methods
see the same resampled frames), recomputes F1 from summed TP/FP/FN and the mean
per-frame WE_acc, and reports B - A with a 95% percentile CI. A CI that contains
0 means the difference is within noise.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def frames(exp: str) -> pd.DataFrame:
    return pd.read_csv(REPO / "runs" / exp / "eval" / exp / "frames.csv", index_col="image")


def f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    return 100 * 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a = frames(args.a)
    for exp in args.b:
        b = frames(exp).loc[a.index]
        idx = np.random.default_rng(args.seed).integers(0, len(a), size=(args.n, len(a)))
        stats = {}
        for name, df in (("a", a), ("b", b)):
            tp, fp, fn = (df[c].to_numpy()[idx].sum(1) for c in ("TP", "FP", "FN"))
            stats[name] = {"F1": f1(tp, fp, fn), "mu": np.nanmean(df["WE_acc"].to_numpy()[idx], axis=1)}
        print(f"{exp} - {args.a} ({len(a)} frames, {args.n} resamples)")
        for m in ("F1", "mu"):
            d = stats["b"][m] - stats["a"][m]
            lo, hi = np.percentile(d, [2.5, 97.5])
            verdict = "within noise" if lo <= 0 <= hi else "significant"
            print(f"  {m:>3}: {np.mean(d):+.1f} (95% CI {lo:+.1f} to {hi:+.1f}) -> {verdict}")


if __name__ == "__main__":
    main()
