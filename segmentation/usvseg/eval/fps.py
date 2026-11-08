"""End-to-end inference speed of Segmenter.segment(): RGB uint8 image in memory ->
Segmentation (labels + confidence) at the original resolution.

Includes pre-processing (resize, normalisation, host->device) and post-processing
(upsampling, argmax / mask merging, confidence map, device->host). Excludes disk I/O. Batch 1,
50 warm-up iterations, 200 timed, cycling through the first --images val images.

    .venv/bin/python -m usvseg.eval.fps --model segformer --weights runs/m1_segformer_b2/checkpoint
    .venv/bin/python -m usvseg.eval.fps --model yolo --weights runs/m2_yolo26s_seg/weights/best.pt
    third_party/eWaSR/.venv/bin/python -m usvseg.eval.fps --model ewasr

Appends one row to results/fps.csv.
"""

import argparse
import csv
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

from common.interfaces import StereoFrame
from usvseg.data.lars import Lars
from usvseg.eval.run_lars_eval import git_commit

REPO = Path(__file__).resolve().parents[2]
FPS_CSV = REPO / "results" / "fps.csv"
COLUMNS = ["date", "git_commit", "model", "weights", "input_res", "image_res", "warmup", "iters",
           "mean_ms", "p50_ms", "p95_ms", "fps", "hardware", "notes"]


def build(model: str, weights: Path | None, args):
    """Returns (Segmenter, input_res description)."""
    if model == "ewasr":
        from usvseg.models.ewasr import DEFAULT_WEIGHTS, EwasrSegmenter
        return EwasrSegmenter(weights or DEFAULT_WEIGHTS), "512x384"
    if model == "segformer":
        from usvseg.models.segformer import SegformerSegmenter
        scale = tuple(int(v) for v in args.scale.split("x"))
        return SegformerSegmenter(weights, scale=scale), f"keep-ratio {args.scale}"
    if model == "yolo":
        from usvseg.models.yolo_seg import YoloSegmenter
        return YoloSegmenter(weights, imgsz=args.imgsz, conf=args.conf), f"imgsz {args.imgsz}"
    raise ValueError(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["ewasr", "segformer", "yolo"])
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--scale", default="2048x1024")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--images", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    samples = Lars("val").samples[: args.images]
    frames = [StereoFrame(left=s.image(), right=None, K=None, baseline_m=None, timestamp=0.0, frame_id=s.name)
              for s in samples]
    seg, input_res = build(args.model, args.weights, args)

    for i in range(args.warmup):
        seg.segment(frames[i % len(frames)])
    times = []
    for i in range(args.iters):
        frame = frames[i % len(frames)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        seg.segment(frame)  # returns numpy arrays, so the GPU work is finished
        torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - t0))

    t = np.array(times)
    hw = sorted({f"{f.left.shape[1]}x{f.left.shape[0]}" for f in frames})
    row = {
        "date": date.today().isoformat(),
        "git_commit": git_commit(),
        "model": args.model,
        "weights": str(args.weights or "default"),
        "input_res": input_res,
        "image_res": " ".join(hw),
        "warmup": args.warmup,
        "iters": args.iters,
        "mean_ms": round(float(t.mean()), 2),
        "p50_ms": round(float(np.percentile(t, 50)), 2),
        "p95_ms": round(float(np.percentile(t, 95)), 2),
        "fps": round(1000 / float(t.mean()), 1),
        "hardware": torch.cuda.get_device_name(0),
        "notes": args.notes,
    }
    print(row)
    if args.no_log:
        return
    new = not FPS_CSV.exists()
    with FPS_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"appended to {FPS_CSV}")


if __name__ == "__main__":
    main()
