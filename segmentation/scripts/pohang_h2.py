"""H2 prototype on Pohang: segmentation + stereo depth -> water plane -> obstacles below it.

Input: data/interim/pohang/<seq>/ from scripts/pohang_stereo_depth.py (rectified left
image and depth from the stereo team's code, unchanged).

    python scripts/pohang_h2.py --checkpoint runs/m1_segformer_b2_e80/checkpoint

Per frame:
1. SegFormer on the rectified left image (RGB).
2. Back-project valid depth to 3D camera coordinates (OpenCV: x right, y down, z forward).
3. RANSAC plane fit on WATER pixels with valid depth within --max-range; the normal
   must be within 30 deg of the camera's "up" (-y), so a quay wall cannot win.
4. Signed height above the plane for every valid pixel (camera side positive).
5. Each OBSTACLE connected component with enough valid depth pixels is flagged as a
   reflection candidate when most of its 3D points lie more than --margin below the plane.

No ground truth exists for Pohang, so this reports plausibility checks (camera height
above water should be stable across frames) and counts, plus overlays in
results/figures/<out name>/ (gitignored).
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from common.interfaces import OBSTACLE, WATER, StereoFrame
from usvseg.models.segformer import SegformerSegmenter
from usvseg.viz.overlay import overlay, side_by_side

REPO = Path(__file__).resolve().parents[1]


def backproject(depth: np.ndarray, P: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = P[0, 0], P[1, 1], P[0, 2], P[1, 2]
    v, u = np.indices(depth.shape)
    return np.stack([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], -1)


def fit_plane(pts: np.ndarray, thresh: float, iters: int, rng: np.random.Generator,
              max_tilt_deg: float = 30.0) -> tuple[np.ndarray, float, float] | None:
    """RANSAC plane n.p + d = 0, n oriented so the camera (origin) has n.0 + d > 0."""
    best, best_inl = None, 0
    up = np.array([0.0, -1.0, 0.0])
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if abs(n @ up) < np.cos(np.radians(max_tilt_deg)):
            continue
        d = -n @ a
        inl = np.abs(pts @ n + d) < thresh
        if inl.sum() > best_inl:
            best, best_inl = (n, d), inl.sum()
    if best is None:
        return None
    # least-squares refit on inliers
    n, d = best
    inl = pts[np.abs(pts @ n + d) < thresh]
    centroid = inl.mean(0)
    n = np.linalg.svd(inl - centroid)[2][-1]
    d = -n @ centroid
    if d < 0:
        n, d = -n, -d
    return n, d, best_inl / len(pts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=REPO / "data" / "interim" / "pohang" / "pohang00_pier")
    ap.add_argument("--max-range", type=float, default=40.0, help="m, stereo noise grows with depth^2")
    ap.add_argument("--thresh", type=float, default=0.2, help="m, RANSAC inlier distance")
    ap.add_argument("--margin", type=float, default=0.5, help="m below the plane to call a point 'under water'")
    ap.add_argument("--min-points", type=int, default=50, help="valid depth pixels needed per component")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "h2_pohang")
    args = ap.parse_args()

    cam = json.loads((args.data / "rectified_camera.json").read_text())
    P = np.array(cam["P1"])
    seg = SegformerSegmenter(args.checkpoint)
    rng = np.random.default_rng(args.seed)
    fig_dir = REPO / "results" / "figures" / args.out.name
    fig_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for fdir in sorted(p for p in args.data.iterdir() if p.is_dir()):
        img = cv2.imread(str(fdir / "left_rectified.png"))[..., ::-1].copy()  # BGR -> RGB
        depth = np.load(fdir / "depth.npy")
        labels = seg.segment(StereoFrame(left=img, right=None, K=P[:, :3], baseline_m=cam["baseline_m"],
                                         timestamp=0.0, frame_id=fdir.name)).labels
        xyz = backproject(depth, P)
        valid = (depth > 0) & (depth < args.max_range)
        water_pts = xyz[valid & (labels == WATER)]
        row = {"frame": fdir.name, "water_px": int((labels == WATER).sum()),
               "water_depth_px": len(water_pts),
               "water_depth_cov_%": round(100 * len(water_pts) / max((labels == WATER).sum(), 1), 1)}
        plane = fit_plane(water_pts, args.thresh, 500, rng) if len(water_pts) >= 200 else None
        flagged_mask = np.zeros(labels.shape, bool)
        if plane is None:
            row.update({"cam_height_m": "", "inlier_%": "", "obst_components": 0, "checked": 0, "flagged": 0,
                        "obst_px_below_%": ""})
        else:
            n, d, inl = plane
            height = xyz @ n + d  # signed height above the water plane, camera side positive
            obst_valid = valid & (labels == OBSTACLE)
            below = obst_valid & (height < -args.margin)
            ncomp, comp = cv2.connectedComponents((labels == OBSTACLE).astype(np.uint8), connectivity=8)
            checked = flagged = 0
            for c in range(1, ncomp):
                m = comp == c
                pts = m & valid
                if pts.sum() < args.min_points:
                    continue
                checked += 1
                if (height[pts] < -args.margin).mean() > 0.5:
                    flagged += 1
                    flagged_mask |= m
            row.update({"cam_height_m": round(float(d), 2), "inlier_%": round(100 * inl, 1),
                        "obst_components": ncomp - 1, "checked": checked, "flagged": flagged,
                        "obst_px_below_%": round(100 * below.sum() / max(obst_valid.sum(), 1), 1)})
        rows.append(row)
        print(row, flush=True)

        depth_vis = cv2.imread(str(fdir / "depth_color.png"))[..., ::-1]
        flag_vis = overlay(img, labels)
        flag_vis[flagged_mask] = (255, 0, 0)
        panels = [img, overlay(img, labels), depth_vis, flag_vis]
        titles = ["rectified left", "SegFormer", "stereo depth (valid)",
                  f"below water plane (red), h={row['cam_height_m']} m"]
        cv2.imwrite(str(fig_dir / f"{fdir.name}.jpg"), side_by_side(panels, titles, height=300)[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    with (args.out / "frames.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    print(f"{len(rows)} frames -> {args.out / 'frames.csv'}, figures -> {fig_dir}")


if __name__ == "__main__":
    main()
