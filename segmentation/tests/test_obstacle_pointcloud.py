from pathlib import Path
import sys

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from obstacle_pointcloud import backproject_obstacle_depth, extract_valid_obstacle_depth


def test_extract_valid_obstacle_depth_filters_depth_and_semantic_classes():
    depth = np.array(
        [[0.0, -1.0, np.nan, np.inf, 39.9, 40.0, 3.0, 2.0]],
        dtype=np.float32,
    )
    labels = np.array([[0, 0, 0, 0, 0, 0, 1, 2]], dtype=np.uint8)

    obstacle_mask, obstacle_depth = extract_valid_obstacle_depth(depth, labels)

    assert obstacle_mask.tolist() == [[False, False, False, False, True, False, False, False]]
    np.testing.assert_allclose(obstacle_depth, [[0.0, 0.0, 0.0, 0.0, 39.9, 0.0, 0.0, 0.0]])


def test_extract_valid_obstacle_depth_rejects_mismatched_shapes_and_invalid_limit():
    with pytest.raises(ValueError, match="dimensions differ"):
        extract_valid_obstacle_depth(np.zeros((2, 2)), np.zeros((2, 1)))

    with pytest.raises(ValueError, match="finite positive"):
        extract_valid_obstacle_depth(np.zeros((1, 1)), np.zeros((1, 1)), np.inf)


def test_backproject_obstacle_depth_returns_camera_xyz_points():
    depth = np.array([[0.0, 2.0], [4.0, 0.0]], dtype=np.float32)
    intrinsics = np.array([[2.0, 0.0, 1.0], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]])

    points = backproject_obstacle_depth(depth, intrinsics)

    np.testing.assert_allclose(points, [[0.0, -0.5, 2.0], [-2.0, 0.0, 4.0]])


def test_backproject_obstacle_depth_returns_empty_array_when_no_depth_is_valid():
    points = backproject_obstacle_depth(
        np.array([[0.0, 40.0, np.nan]], dtype=np.float32),
        np.eye(3),
    )

    assert points.shape == (0, 3)
