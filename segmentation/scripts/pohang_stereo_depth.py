"""Run the stereo team's depth code (stereo/stereo_depth.py, unchanged) on a subset of Pohang frames.

Needs OpenCV contrib (cv2.ximgproc), so it runs in its own venv:
    python3 -m venv .venv-stereo
    .venv-stereo/bin/pip install numpy==2.5.4 opencv-contrib-python-headless==5.0.0.93
    .venv-stereo/bin/python scripts/pohang_stereo_depth.py --stereo-script ../stereo/stereo_depth.py --every 10

The stereo module is loaded from a path, so it is used exactly as committed by the
stereo team. Per frame it writes left_rectified.png, depth.npy, disparity.npy, ...
to data/interim/pohang/<seq>/<frame>/; the rectified projection matrix P1 (needed to
back-project depth to 3D) goes to rectified_camera.json.
"""

import argparse
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stereo-script", type=Path, default=REPO.parent / "stereo" / "stereo_depth.py")
    ap.add_argument("--seq", type=Path, default=REPO / "data" / "raw" / "pohang" / "pohang00_pier")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--every", type=int, default=10, help="use every N-th frame")
    ap.add_argument("--frames", nargs="*", default=None, help="explicit frame ids instead of --every")
    ap.add_argument("--num-disparities", type=int, default=256, help="SGBM search range (stereo code default 256)")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("stereo_depth", args.stereo_script)
    sd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sd)  # computes the rectification maps at import

    out = args.out or REPO / "data" / "interim" / "pohang" / args.seq.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "rectified_camera.json").write_text(json.dumps({
        "P1": sd.P1.tolist(), "Q": sd.Q.tolist(), "baseline_m": float(sd.baseline),
        "image_size": list(sd.IMAGE_SIZE)}, indent=1))

    left = sorted((args.seq / "stereo" / "left_images").glob("*.png"))
    left = [p for p in left if p.stem in args.frames] if args.frames else left[:: args.every]
    for lp in left:
        rp = args.seq / "stereo" / "right_images" / lp.name
        sd.compute_depth(lp, rp, out / lp.stem, num_disparities=args.num_disparities)
    print(f"{len(left)} frames -> {out}")


if __name__ == "__main__":
    main()
