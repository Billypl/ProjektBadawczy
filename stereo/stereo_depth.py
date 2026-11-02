"""
tested using data from https://drive.google.com/file/d/11OIxIHHd7w279Ct8QaGp6VCr1WEihb5W/view

python .\stereo_depth.py --left .\dataset\pohang00_pier\stereo\left_images\ --right .\dataset\pohang00_pier\stereo\right_images\ --output results

"""

import cv2
import numpy as np
from pathlib import Path
import argparse


# --------------------------------------------------
# Camera calibration
# --------------------------------------------------

K1 = np.array([
    [1814.5061420730431, 0, 1026.1977130691339],
    [0, 1814.5061420730431, 524.7984412358129],
    [0, 0, 1]
], dtype=np.float64)

K2 = np.array([
    [1812.26306845315, 0, 1041.300936202379],
    [0, 1812.26306845315, 519.0722308029441],
    [0, 0, 1]
], dtype=np.float64)

D1 = np.array([
    -0.08229280826733834,
    0.1036602731851739,
    -0.0034940032414059268,
    0.0008401400230805731,
    0.09370029132155965
], dtype=np.float64)

D2 = np.array([
    -0.08022652691881974,
    0.0930632046901962,
    -0.002945409583790114,
    0.0011427467119549529,
    0.1167562399019816
], dtype=np.float64)


# Quaternions assumed to be [x, y, z, w]
# Assumed camera-to-world poses.

Q1 = [0.5045187477289094,
      0.4927116633560035,
      0.5032030494153246,
      0.4994824732081517]

Q2 = [0.4995715149989711,
      0.4932107734020534,
      0.5032303654146303,
      0.5039153041200861]

C1 = np.array([
    5.036254972834461,
    -0.4410102127584988,
    -1.771000242852925
], dtype=np.float64)

C2 = np.array([
    5.010057439449895,
    0.4023181598949916,
    -1.7616651319571297
], dtype=np.float64)


IMAGE_SIZE = (2048, 1080)


# --------------------------------------------------
# Quaternion conversion
# --------------------------------------------------

def quaternion_to_rotation(q):
    """
    Quaternion order: [qx, qy, qz, qw]
    Returns sensor-to-AHRS rotation matrix.
    """
    x, y, z, w = np.asarray(q, dtype=np.float64)

    norm = np.linalg.norm([x, y, z, w])
    x, y, z, w = x/norm, y/norm, z/norm, w/norm

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])

    return R


# --------------------------------------------------
# Calculate relative stereo pose
# --------------------------------------------------

R1 = quaternion_to_rotation(Q1)
R2 = quaternion_to_rotation(Q2)

# Camera-to-world poses:
# X_world = R_camera_to_world * X_camera + C

# Transform from left camera coordinates
# to right camera coordinates.

R = R2.T @ R1
T = (R2.T @ (C1 - C2)).reshape(3, 1)

baseline = np.linalg.norm(C2 - C1)

print("Stereo baseline:", baseline)
print("Relative translation:", T)
print("Relative rotation:\n", R)


# --------------------------------------------------
# Stereo rectification
# --------------------------------------------------

R_rect1, R_rect2, P1, P2, Q, roi1, roi2 = (
    cv2.stereoRectify(
        K1, D1,
        K2, D2,
        IMAGE_SIZE,
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )
)

map1x, map1y = cv2.initUndistortRectifyMap(
    K1, D1, R_rect1, P1,
    IMAGE_SIZE,
    cv2.CV_32FC1
)

map2x, map2y = cv2.initUndistortRectifyMap(
    K2, D2, R_rect2, P2,
    IMAGE_SIZE,
    cv2.CV_32FC1
)


# --------------------------------------------------
# Process one stereo frame pair
# --------------------------------------------------

def compute_depth(left_path, right_path, output_dir,
                  num_disparities=256,
                  block_size=5):

    left = cv2.imread(str(left_path))
    right = cv2.imread(str(right_path))

    if left is None or right is None:
        raise RuntimeError("Could not load stereo images.")

    if left.shape[:2] != (1080, 2048):
        raise ValueError(
            f"Unexpected left image size: {left.shape[1]}x{left.shape[0]}"
        )

    if right.shape != left.shape:
        raise ValueError("Left and right images must have identical sizes.")

    # Rectify images
    left_rect = cv2.remap(
        left, map1x, map1y, cv2.INTER_LINEAR
    )

    right_rect = cv2.remap(
        right, map2x, map2y, cv2.INTER_LINEAR
    )

    # Save rectified images for inspection
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(output_dir / "left_rectified.png"), left_rect
    )

    cv2.imwrite(
        str(output_dir / "right_rectified.png"), right_rect
    )

    # Grayscale
    grayL = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    grayL = clahe.apply(grayL)
    grayR = clahe.apply(grayR)

    # Stereo matching
    block_size = max(3, block_size)
    if block_size % 2 == 0:
        block_size += 1

    num_disparities = max(
        16, (num_disparities // 16) * 16
    )

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size**2,
        P2=32 * block_size**2,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=5,
        speckleWindowSize=150,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(
        left_matcher
    )

    dispL = left_matcher.compute(grayL, grayR)
    dispR = right_matcher.compute(grayR, grayL)

    # WLS filtering
    wls = cv2.ximgproc.createDisparityWLSFilter(
        matcher_left=left_matcher
    )

    wls.setLambda(8000)
    wls.setSigmaColor(1.5)

    filtered = wls.filter(
        dispL, left_rect, None, dispR
    )

    confidence = wls.getConfidenceMap()

    disparity = filtered.astype(np.float32) / 16.0

    # Valid disparity
    valid = (
        (disparity > 0) &
        (confidence > 50)
    )

    disparity[~valid] = 0

    # Depth reconstruction using Q matrix
    points_3D = cv2.reprojectImageTo3D(
        disparity, Q
    )

    depth = points_3D[:, :, 2]

    valid &= np.isfinite(depth) & (depth > 0)

    depth[~valid] = 0

    # Visualization of disparity
    disparity_vis = np.zeros(
        disparity.shape, dtype=np.uint8
    )

    if np.any(valid):

        lo, hi = np.percentile(
            disparity[valid], [2, 98]
        )

        if hi > lo:

            normalized = np.clip(
                (disparity - lo) / (hi - lo),
                0, 1
            )

            disparity_vis[valid] = (
                normalized[valid] * 255
            ).astype(np.uint8)

    depth_color = cv2.applyColorMap(
        disparity_vis,
        cv2.COLORMAP_TURBO
    )

    depth_color[~valid] = 0

    # Save outputs
    cv2.imwrite(
        str(output_dir / "disparity.png"),
        disparity_vis
    )

    cv2.imwrite(
        str(output_dir / "depth_color.png"),
        depth_color
    )

    cv2.imwrite(
        str(output_dir / "confidence.png"),
        confidence.astype(np.uint8)
    )

    np.save(
        output_dir / "disparity.npy",
        disparity
    )

    np.save(
        output_dir / "depth.npy",
        depth
    )

    print(
        f"{left_path.name}: "
        f"{np.mean(valid)*100:.1f}% valid pixels"
    )


# --------------------------------------------------
# Process all matching frames
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", default="depth_results")

    parser.add_argument(
        "--num-disparities",
        type=int,
        default=256
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=5
    )

    args = parser.parse_args()

    left_dir = Path(args.left)
    right_dir = Path(args.right)

    left_frames = {
        p.name: p for p in left_dir.glob("*.png")
    }

    right_frames = {
        p.name: p for p in right_dir.glob("*.png")
    }

    common = sorted(
        set(left_frames) & set(right_frames)
    )

    if not common:
        raise RuntimeError("No matching stereo frames.")

    print(f"Found {len(common)} frame pairs.")

    for filename in common:

        compute_depth(
            left_frames[filename],
            right_frames[filename],
            Path(args.output) / Path(filename).stem,
            args.num_disparities,
            args.block_size
        )


if __name__ == "__main__":
    main()
