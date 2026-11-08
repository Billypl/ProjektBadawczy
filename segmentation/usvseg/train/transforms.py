"""Joint image/mask augmentations for semantic segmentation training.

Follows the MaCVi LaRS starter kit (mmsegmentation-macvi): keep-ratio resize to a
base scale with a random ratio, random crop with cat_max_ratio, horizontal flip,
photometric jitter. Adds a small rotation (boat roll). No vertical flip: sky is
always on top and the model should use that prior.

Masks are always resampled with nearest neighbour and padded with IGNORE.
"""

import cv2
import numpy as np
import torch
from torchvision.transforms import ColorJitter

from common.interfaces import IGNORE

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def keep_ratio_scale(hw: tuple[int, int], scale: tuple[int, int]) -> float:
    """mmseg keep_ratio rule: long side <= max(scale), short side <= min(scale)."""
    h, w = hw
    return min(max(scale) / max(h, w), min(scale) / min(h, w))


def resize_pair(img: np.ndarray, mask: np.ndarray | None, factor: float):
    h, w = img.shape[:2]
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return img, mask


def normalize(img: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 RGB -> 3xHxW float tensor, ImageNet normalisation."""
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


class TrainTransform:
    def __init__(self, base_scale=(2048, 1024), ratio_range=(0.5, 2.0), crop_hw=(512, 1024),
                 cat_max_ratio=0.75, flip_prob=0.5, rotate_deg=5.0, rotate_prob=0.5,
                 color_jitter=(0.3, 0.3, 0.3, 0.05)):
        self.base_scale = tuple(base_scale)
        self.ratio_range = tuple(ratio_range)
        self.crop_hw = tuple(crop_hw)
        self.cat_max_ratio = cat_max_ratio
        self.flip_prob = flip_prob
        self.rotate_deg = rotate_deg
        self.rotate_prob = rotate_prob
        self.jitter = ColorJitter(*color_jitter)

    def __call__(self, img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
        ratio = rng.uniform(*self.ratio_range)
        img, mask = resize_pair(img, mask, keep_ratio_scale(img.shape[:2], self.base_scale) * ratio)

        if self.rotate_deg and rng.random() < self.rotate_prob:
            img, mask = self._rotate(img, mask, rng.uniform(-self.rotate_deg, self.rotate_deg))

        img, mask = self._crop(img, mask, rng)

        if rng.random() < self.flip_prob:
            img, mask = img[:, ::-1], mask[:, ::-1]

        x = self.jitter(torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1))
        x = normalize(x.permute(1, 2, 0).numpy())
        return x, torch.from_numpy(np.ascontiguousarray(mask)).long()

    @staticmethod
    def _rotate(img, mask, deg):
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=IGNORE)
        return img, mask

    def _crop(self, img, mask, rng):
        ch, cw = self.crop_hw
        h, w = mask.shape
        if h < ch or w < cw:  # pad to crop size
            ph, pw = max(ch - h, 0), max(cw - w, 0)
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=IGNORE)
            h, w = mask.shape
        # mmseg cat_max_ratio: retry so that no single class dominates the crop
        for _ in range(10):
            y, x = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
            crop = mask[y:y + ch, x:x + cw]
            _, counts = np.unique(crop[crop != IGNORE], return_counts=True)
            if len(counts) > 1 and counts.max() / counts.sum() < self.cat_max_ratio:
                break
        return img[y:y + ch, x:x + cw], mask[y:y + ch, x:x + cw]
