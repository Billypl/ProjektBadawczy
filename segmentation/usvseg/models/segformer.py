"""SegFormer (M1) via HF transformers: model construction and inference.

Encoder weights nvidia/mit-b2 (ImageNet-1k), NVIDIA Source Code License
(non-commercial research use).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from common.interfaces import IGNORE, Segmentation, StereoFrame
from usvseg.data.taxonomy import SEMANTIC_CLASSES
from usvseg.train.transforms import keep_ratio_scale, normalize, resize_pair


def build_segformer(encoder: str = "nvidia/mit-b2") -> SegformerForSemanticSegmentation:
    return SegformerForSemanticSegmentation.from_pretrained(
        encoder,
        num_labels=len(SEMANTIC_CLASSES),
        id2label={k: v for k, v in SEMANTIC_CLASSES.items()},
        label2id={v: k for k, v in SEMANTIC_CLASSES.items()},
        semantic_loss_ignore_index=IGNORE,
    )


class SegformerSegmenter:
    """Keep-ratio resize to `scale`, whole-image inference, logits upsampled to the original size."""

    def __init__(self, checkpoint: Path, scale: tuple[int, int] = (2048, 1024), device: str = "cuda",
                 amp: bool = True):
        self.model = SegformerForSemanticSegmentation.from_pretrained(checkpoint).eval().to(device)
        self.scale = tuple(scale)
        self.device = torch.device(device)
        self.amp = amp

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        img, _ = resize_pair(image, None, keep_ratio_scale((h, w), self.scale))
        x = normalize(img)[None].to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            logits = self.model(pixel_values=x).logits
        logits = F.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False)
        conf, labels = logits.softmax(1)[0].max(0)
        return labels.byte().cpu().numpy(), conf.cpu().numpy()

    def segment(self, frame: StereoFrame) -> Segmentation:
        labels, conf = self.predict(frame.left)
        return Segmentation(labels=labels, confidence=conf)
