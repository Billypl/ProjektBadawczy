"""Shared contract between segmentation (Billy) and stereo/depth (Filip, Hania).

All arrays are in the geometry of the rectified left image.
Camera frame follows OpenCV: x right, y down, z forward, units in metres.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

# Class ids follow LaRS / MaSTr semantic masks.
OBSTACLE, WATER, SKY, IGNORE = 0, 1, 2, 255


@dataclass
class StereoFrame:
    left: np.ndarray        # HxWx3 uint8, RGB, after rectification
    right: np.ndarray | None
    K: np.ndarray | None    # 3x3 intrinsics after rectification
    baseline_m: float | None
    timestamp: float
    frame_id: str


@dataclass
class Instance:
    cls: int                # dataset-specific object class id
    mask: np.ndarray        # HxW bool
    score: float


@dataclass
class Segmentation:
    labels: np.ndarray      # HxW uint8, values in {0,1,2,255}
    confidence: np.ndarray  # HxW float32 in [0,1]
    instances: list[Instance] = field(default_factory=list)


class Segmenter(Protocol):
    def segment(self, frame: StereoFrame) -> Segmentation: ...
