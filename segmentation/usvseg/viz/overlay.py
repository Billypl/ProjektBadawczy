"""Mask overlays and side-by-side comparison panels."""

import cv2
import numpy as np

from usvseg.data.taxonomy import labels_to_color


def overlay(image: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend evaluator colours onto an RGB image; ignore pixels stay untouched."""
    color = labels_to_color(labels)
    out = image.copy()
    valid = labels != 255
    out[valid] = (image[valid] * (1 - alpha) + color[valid] * alpha).astype(np.uint8)
    return out


def side_by_side(panels: list[np.ndarray], titles: list[str], height: int = 360) -> np.ndarray:
    """Resize RGB panels to a common height and stack horizontally with titles."""
    resized = []
    for img, title in zip(panels, titles, strict=True):
        w = round(img.shape[1] * height / img.shape[0])
        p = cv2.resize(img, (w, height), interpolation=cv2.INTER_AREA).copy()
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(p, (0, 0), (tw + 12, th + 12), (0, 0, 0), -1)
        cv2.putText(p, title, (6, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        resized.append(p)
    return np.concatenate(resized, axis=1)
