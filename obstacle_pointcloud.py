# Obstacle point cloud from stereo depth and semantic segmentation.
#
# End-to-end input:
# - A matching pair of unrectified left/right stereo images.
# - A configured segmenter, which predicts labels from the rectified left RGB image.
# - Stereo calibration and the rectified intrinsics produced from it.
#
# Processing order:
# 1. [DONE] Compute depth from the complete left/right pair with
#    stereo/stereo_depth.py, then segment its rectified left RGB image.
#    Do not mask either stereo image before computing depth.
# 2. [DONE] Confirm that depth and label arrays have identical height and
#    width; this pipeline segments the same rectified left image used for depth.
# 3. [DONE] Build a validity mask from finite depth values greater than zero
#    and less than the approximate 40 m range limit. Zero means missing depth.
# 4. [DONE] Keep only valid pixels labeled OBSTACLE (class id 0). Discard SKY
#    (class id 2) and IGNORE (class id 255). WATER (class id 1) is not part of
#    this obstacle point cloud; it may be selected separately for water-plane
#    processing.
# 5. [DONE] Back-project each retained pixel (u, v) with depth Z to camera
#    coordinates in metres:
#       X = (u - cx) * Z / fx
#       Y = (v - cy) * Z / fy
#       Z = depth[v, u]
#    OpenCV camera convention: x right, y down, z forward. Coordinates are
#    relative to the rectified left camera.
# 6. [DONE] Save the obstacle points as an N-by-3 array in obstacle_points.npy.
#
# Calibration note:
# - The dataset JSON contains raw camera intrinsics/extrinsics, not rectified
#   intrinsics. stereoRectify derives the effective rectified camera matrix;
#   its left 3x3 block is used for pixels in the rectified depth map.
# - The resulting points use the rectified left-camera origin and axes. No
#   translation to the stereo midpoint is applied.

import argparse
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from types import ModuleType

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
SEGMENTATION_DIR = ROOT_DIR / "segmentation"

# Allow this root-level script to import the sibling segmentation project.
if str(SEGMENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENTATION_DIR))

from common.interfaces import OBSTACLE, Segmentation, Segmenter, StereoFrame

MAX_DEPTH_M = 40.0


@lru_cache(maxsize=1)
def _load_stereo_module() -> ModuleType:
    """Load stereo calibration once; importing that module also builds rectification maps."""
    # Anchor the sibling stereo folder to this file, not to the checkout folder's name.
    stereo_script = ROOT_DIR / "stereo" / "stereo_depth.py"
    spec = spec_from_file_location("zpb_stereo_depth", stereo_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load stereo depth module: {stereo_script}")

    stereo_module = module_from_spec(spec)
    spec.loader.exec_module(stereo_module)
    return stereo_module


def compute_depth_and_segment(
    left_path: Path,
    right_path: Path,
    output_dir: Path,
    segmenter: Segmenter,
    *,
    timestamp: float,
    K: np.ndarray | None = None,
    baseline_m: float | None = None,
) -> tuple[np.ndarray, Segmentation]:
    """Compute depth for a raw stereo pair, then segment the resulting rectified left image.

    The stereo script writes depth.npy and left_rectified.png into output_dir.
    The caller supplies an initialized segmenter and the frame metadata it needs.
    """
    _load_stereo_module().compute_depth(left_path, right_path, output_dir)

    depth, segmentation = load_depth_and_segment(
        output_dir / "depth.npy",
        output_dir / "left_rectified.png",
        segmenter,
        timestamp=timestamp,
        K=K,
        baseline_m=baseline_m,
    )
    if depth.shape != segmentation.labels.shape:
        raise ValueError(
            "Depth and segmentation dimensions differ: "
            f"{depth.shape} != {segmentation.labels.shape}"
        )
    return depth, segmentation


def load_depth_and_segment(
    depth_path: Path,
    rectified_left_path: Path,
    segmenter: Segmenter,
    *,
    timestamp: float,
    K: np.ndarray | None = None,
    baseline_m: float | None = None,
) -> tuple[np.ndarray, Segmentation]:
    """Segment an already computed depth map and its matching rectified left image.

    OpenCV loads the image as BGR, while the segmentation interface expects RGB.
    The right image is omitted because stereo matching has already used it to
    produce the depth map.
    """
    depth = np.load(depth_path)
    left_bgr = cv2.imread(str(rectified_left_path), cv2.IMREAD_COLOR)
    if left_bgr is None:
        raise FileNotFoundError(f"Could not read rectified left image: {rectified_left_path}")

    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    frame = StereoFrame(
        left=left_rgb,
        right=None,
        K=K,
        baseline_m=baseline_m,
        timestamp=timestamp,
        frame_id=rectified_left_path.stem,
    )
    segmentation = segmenter.segment(frame)
    return depth, segmentation


def extract_valid_obstacle_depth(
    depth: np.ndarray,
    labels: np.ndarray,
    max_depth_m: float = MAX_DEPTH_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a mask and depth map containing only valid, in-range obstacle pixels."""
    if depth.ndim != 2 or labels.ndim != 2:
        raise ValueError(
            f"Depth and labels must be 2D arrays; got {depth.ndim}D and {labels.ndim}D"
        )
    if depth.shape != labels.shape:
        raise ValueError(f"Depth and label dimensions differ: {depth.shape} != {labels.shape}")
    if not np.isfinite(max_depth_m) or max_depth_m <= 0:
        raise ValueError(f"max_depth_m must be a finite positive value, got {max_depth_m}")

    valid_depth = np.isfinite(depth) & (depth > 0) & (depth < max_depth_m)
    obstacle_mask = valid_depth & (labels == OBSTACLE)
    obstacle_depth = np.where(obstacle_mask, depth, 0)
    return obstacle_mask, obstacle_depth


def backproject_obstacle_depth(
    obstacle_depth: np.ndarray,
    rectified_intrinsics: np.ndarray,
) -> np.ndarray:
    """Convert the filtered depth map to N x 3 points in the rectified left-camera frame."""
    if obstacle_depth.ndim != 2:
        raise ValueError(f"Obstacle depth must be 2D, got shape {obstacle_depth.shape}")
    if rectified_intrinsics.shape != (3, 3):
        raise ValueError(
            "Rectified intrinsics must be a 3x3 matrix, "
            f"got shape {rectified_intrinsics.shape}"
        )
    if not np.all(np.isfinite(rectified_intrinsics)):
        raise ValueError("Rectified intrinsics must contain only finite values")

    fx, fy = rectified_intrinsics[0, 0], rectified_intrinsics[1, 1]
    cx, cy = rectified_intrinsics[0, 2], rectified_intrinsics[1, 2]
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Rectified focal lengths must be positive, got fx={fx}, fy={fy}")

    valid = np.isfinite(obstacle_depth) & (obstacle_depth > 0) & (obstacle_depth < MAX_DEPTH_M)
    v, u = np.nonzero(valid)
    z = obstacle_depth[v, u]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.column_stack((x, y, z))


def _create_segmenter(checkpoint: Path, device: str) -> Segmenter:
    from usvseg.models.segformer import SegformerSegmenter

    return SegformerSegmenter(checkpoint, device=device, amp=device.startswith("cuda"))


def _process_and_save_frame(
    left_path: Path,
    right_path: Path,
    output_dir: Path,
    segmenter: Segmenter,
    *,
    timestamp: float,
) -> dict[str, object]:
    from usvseg.viz.overlay import overlay

    output_dir.mkdir(parents=True, exist_ok=True)
    depth, segmentation = compute_depth_and_segment(
        left_path,
        right_path,
        output_dir,
        segmenter,
        timestamp=timestamp,
    )

    labels_path = output_dir / "segmentation_labels.png"
    np.save(output_dir / "segmentation_labels.npy", segmentation.labels)
    np.save(output_dir / "segmentation_confidence.npy", segmentation.confidence)
    if not cv2.imwrite(str(labels_path), segmentation.labels):
        raise OSError(f"Could not write segmentation labels: {labels_path}")

    rectified_path = output_dir / "left_rectified.png"
    rectified_bgr = cv2.imread(str(rectified_path), cv2.IMREAD_COLOR)
    if rectified_bgr is None:
        raise FileNotFoundError(rectified_path)
    rectified_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = overlay(rectified_rgb, segmentation.labels)
    overlay_path = output_dir / "segmentation_overlay.png"
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(overlay_path), overlay_bgr):
        raise OSError(f"Could not write segmentation overlay: {overlay_path}")

    obstacle_mask, obstacle_depth = extract_valid_obstacle_depth(depth, segmentation.labels)
    obstacle_mask_path = output_dir / "obstacle_mask.png"
    if not cv2.imwrite(str(obstacle_mask_path), obstacle_mask.astype(np.uint8) * 255):
        raise OSError(f"Could not write obstacle mask: {obstacle_mask_path}")
    np.save(output_dir / "obstacle_mask.npy", obstacle_mask)
    np.save(output_dir / "obstacle_depth.npy", obstacle_depth)

    # P1's left 3x3 block is the rectified intrinsic matrix used to remap this image.
    rectified_intrinsics = _load_stereo_module().P1[:, :3]
    obstacle_points = backproject_obstacle_depth(obstacle_depth, rectified_intrinsics)
    np.save(output_dir / "obstacle_points.npy", obstacle_points)

    valid_depth = np.isfinite(depth) & (depth > 0) & (depth < MAX_DEPTH_M)
    return {
        "frame_id": left_path.stem,
        "depth_shape": list(depth.shape),
        "labels_shape": list(segmentation.labels.shape),
        "valid_depth_pixels": int(np.count_nonzero(valid_depth)),
        "semantic_obstacle_pixels": int(np.count_nonzero(segmentation.labels == OBSTACLE)),
        "obstacle_pixels": int(np.count_nonzero(obstacle_mask)),
        "obstacle_points": int(len(obstacle_points)),
        "point_coordinate_frame": "rectified_left_camera",
        "output_dir": str(output_dir),
    }


def run_test_frames(
    left_dir: Path,
    right_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    num_frames: int = 3,
    device: str = "cuda",
) -> list[dict[str, object]]:
    """Run the full depth/SegFormer pipeline on the first matching stereo frames."""
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1")
    if not left_dir.is_dir() or not right_dir.is_dir():
        raise NotADirectoryError("left_dir and right_dir must both be directories")

    right_by_name = {path.name: path for path in right_dir.glob("*.png")}
    pairs = [
        (left_path, right_by_name[left_path.name])
        for left_path in sorted(left_dir.glob("*.png"))
        if left_path.name in right_by_name
    ]
    if len(pairs) < num_frames:
        raise ValueError(
            f"Requested {num_frames} matching stereo pairs, found {len(pairs)}"
        )

    segmenter = _create_segmenter(checkpoint, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        _process_and_save_frame(
            left_path,
            right_path,
            output_dir / left_path.stem,
            segmenter,
            timestamp=0.0,
        )
        for left_path, right_path in pairs[:num_frames]
    ]
    (output_dir / "test_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute stereo depth and segment one pair or a directory of frames."
    )
    parser.add_argument(
        "--left", type=Path, required=True, help="Left image path or directory of left frames."
    )
    parser.add_argument(
        "--right", type=Path, required=True, help="Matching right image path or directory of right frames."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SEGMENTATION_DIR / "runs" / "segformer_b2_lars_e80",
        help="SegFormer checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "results" / "obstacle_pointcloud",
        help="Directory for depth and segmentation outputs.",
    )
    parser.add_argument("--device", default="cuda", help="Segmentation device, for example cuda or cpu.")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of matching pairs to process when --left and --right are directories.",
    )
    parser.add_argument(
        "--timestamp",
        type=float,
        default=0.0,
        help="Frame timestamp metadata; SegFormer does not use it.",
    )
    args = parser.parse_args()

    if args.left.is_dir() != args.right.is_dir():
        parser.error("--left and --right must both be files or both be directories")

    if args.left.is_dir():
        summaries = run_test_frames(
            args.left,
            args.right,
            args.checkpoint,
            args.output_dir,
            num_frames=args.num_frames,
            device=args.device,
        )
        print(f"Processed {len(summaries)} pairs; summary: {args.output_dir / 'test_summary.json'}")
        return

    if args.num_frames != 1:
        parser.error("--num-frames can only be used when --left and --right are directories")

    segmenter = _create_segmenter(args.checkpoint, args.device)
    summary = _process_and_save_frame(
        args.left,
        args.right,
        args.output_dir,
        segmenter,
        timestamp=args.timestamp,
    )
    print(f"Processed frame {summary['frame_id']}; results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
