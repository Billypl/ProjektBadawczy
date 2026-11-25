"""Optional post-processing of semantic label maps (off by default everywhere).

remove_small_blobs: small OBSTACLE components surrounded by water are relabelled to
WATER. With shore_dist > 0 only blobs within shore_dist px of a large obstacle region
(shore, pier, big vessel) are removed; with shore_dist = 0 any small blob is removed.
Motivation: river analysis, 37% of SegFormer FP pixels lie < 10 px from the shore.

Areas and distances are in pixels of the label map passed in (original image
resolution when used from the Segmenters).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from common.interfaces import OBSTACLE, WATER


@dataclass(frozen=True)
class BlobFilter:
    max_area: int = 200      # components with area < max_area px are candidates
    shore_dist: int = 10     # px to the nearest large obstacle region; 0 = remove anywhere
    water_frac: float = 0.5  # only blobs whose 1-px border is mostly water (not next to sky)

    def tag(self) -> str:
        return f"pp_a{self.max_area}_d{self.shore_dist}"


def remove_small_blobs(labels: np.ndarray, f: BlobFilter) -> np.ndarray:
    obst = (labels == OBSTACLE).astype(np.uint8)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(obst, connectivity=8)
    if n <= 1:
        return labels
    area = stats[:, cv2.CC_STAT_AREA]
    small = np.zeros(n, bool)
    small[1:] = area[1:] < f.max_area
    if not small.any():
        return labels

    if f.shore_dist > 0:
        large = (comp > 0) & ~small[comp]
        # distance of every pixel to the nearest large obstacle pixel
        dist = cv2.distanceTransform((~large).astype(np.uint8), cv2.DIST_L2, 3)
    out = labels.copy()
    water = labels == WATER
    kernel = np.ones((3, 3), np.uint8)
    for c in np.flatnonzero(small):
        x, y, w, h = stats[c, :4]
        # work in a padded bounding box, the blobs are small
        y0, y1, x0, x1 = max(y - 1, 0), min(y + h + 1, labels.shape[0]), max(x - 1, 0), min(x + w + 1, labels.shape[1])
        m = comp[y0:y1, x0:x1] == c
        ring = cv2.dilate(m.astype(np.uint8), kernel).astype(bool) & ~m
        if not ring.any() or water[y0:y1, x0:x1][ring].mean() < f.water_frac:
            continue
        if f.shore_dist > 0 and dist[y0:y1, x0:x1][m].min() > f.shore_dist:
            continue
        out[y0:y1, x0:x1][m] = WATER
    return out
