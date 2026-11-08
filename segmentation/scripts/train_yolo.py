"""Train YOLO-seg on the converted LaRS dataset.

    python scripts/train_yolo.py --config configs/m2_yolo26s_seg.yaml [--smoke]

Ultralytics writes to runs/<exp>/ (weights/best.pt, weights/last.pt, results.csv).
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 20 images (fraction of train)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    exp = cfg["exp"] + ("_smoke" if args.smoke else "")
    out = REPO / "runs" / exp
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    model = YOLO(str(REPO / "runs" / "weights" / cfg["weights"]))
    model.train(
        data=str(REPO / cfg["data"]),
        epochs=1 if args.smoke else cfg["epochs"],
        fraction=20 / 2345 if args.smoke else 1.0,
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        workers=cfg["workers"],
        seed=cfg["seed"],
        deterministic=True,
        project=str(REPO / "runs"),
        name=exp,
        exist_ok=True,
        plots=True,
        **cfg["augment"],
    )


if __name__ == "__main__":
    main()
