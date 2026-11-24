"""Write predictions in the lars_evaluator semantic format (colour-coded RGB PNG)."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from usvseg.data.taxonomy import labels_to_color


def resize_labels(labels: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a label map to (H, W)."""
    if labels.shape[:2] == hw:
        return labels
    return cv2.resize(labels, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)


def save_prediction(labels: np.ndarray, out_dir: Path, name: str, image_hw: tuple[int, int]) -> Path:
    """Save HxW class ids as an evaluator-coloured PNG at the original image resolution."""
    labels = resize_labels(labels.astype(np.uint8), image_hw)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    Image.fromarray(labels_to_color(labels)).save(path)
    return path
