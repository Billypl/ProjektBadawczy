"""Run the official lars_evaluator on a prediction dir and log the row to results/results.csv.

    python -m usvseg.eval.run_lars_eval --exp m0_ewasr_val --model ewasr_resnet18 \
        --train-data MaSTr1325 --split val --input-res 512x384

Predictions are expected in runs/<exp>/preds/<exp>/<name>.png (colour-coded).
Evaluator outputs (summary.json, frames.csv, ...) land in runs/<exp>/eval/<exp>/.
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import torch
import yaml

from usvseg.data.lars import DEFAULT_ROOT, Lars, _find_dir

REPO = Path(__file__).resolve().parents[2]
EVALUATOR_DIR = REPO / "third_party" / "lars_evaluator"
RESULTS_CSV = REPO / "results" / "results.csv"
COLUMNS = ["date", "git_commit", "exp_name", "model", "train_data", "eval_split", "input_res",
           "F1", "mu", "mIoU", "Q", "fps", "hardware", "notes"]


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def build_config(exp_dir: Path, exp: str, split: str, root: Path) -> Path:
    ds = Lars(split, root)
    ann_dir = _find_dir(root, ds.source_split, "semantic_masks").parent
    subset = exp_dir / f"{split}_image_list.txt"
    subset.write_text("\n".join(s.name for s in ds.samples) + "\n")

    base = yaml.safe_load((EVALUATOR_DIR / "configs" / "v1.0.0" / "lars_val_semantic.yaml").read_text())
    base["PATHS"] = {
        "DATASET_ROOT": str(ann_dir),
        "PREDICTIONS": str(exp_dir / "preds"),
        "RESULTS": str(exp_dir / "eval"),
    }
    base["DATASET"]["SUBSET_LIST"] = str(subset)  # absolute path wins in os.path.join
    cfg = exp_dir / "eval_config.yaml"
    cfg.write_text(yaml.safe_dump(base, sort_keys=False))
    return cfg


def append_result(row: dict) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--split", default="val", choices=["val", "dev"])
    ap.add_argument("--input-res", default="")
    ap.add_argument("--fps", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--no-log", action="store_true", help="do not append to results.csv")
    args = ap.parse_args()

    exp_dir = REPO / "runs" / args.exp
    if not (exp_dir / "preds" / args.exp).is_dir():
        sys.exit(f"missing predictions: {exp_dir / 'preds' / args.exp}")
    cfg = build_config(exp_dir, args.exp, args.split, args.root)

    subprocess.run([sys.executable, "evaluate.py", str(cfg), args.exp, "--workers", "1"],
                   cwd=EVALUATOR_DIR, check=True)
    summary = json.loads((exp_dir / "eval" / args.exp / "summary.json").read_text())
    print(json.dumps({k: summary[k] for k in ("F1", "WE_acc", "mIoU", "Q")}, indent=2))

    if args.no_log:
        return
    append_result({
        "date": date.today().isoformat(),
        "git_commit": git_commit(),
        "exp_name": args.exp,
        "model": args.model,
        "train_data": args.train_data,
        "eval_split": args.split,
        "input_res": args.input_res,
        "F1": round(summary["F1"], 2),
        "mu": round(summary["WE_acc"], 2),
        "mIoU": round(summary["mIoU"], 2),
        "Q": round(summary["Q"], 2),
        "fps": args.fps,
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "notes": args.notes,
    })
    print(f"appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
