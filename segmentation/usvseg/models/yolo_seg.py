"""YOLO26-seg (M2) wrapper. Ultralytics is AGPL-3.0; keep all its imports in this module.

Instance predictions -> semantic map: water first, sky over water, obstacles
(static obstacle + all thing classes) over both. Pixels no instance covers get
`unassigned`: OBSTACLE (cautious, default) or WATER (optimistic).
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from common.interfaces import OBSTACLE, WATER, Instance, Segmentation, StereoFrame
from usvseg.data.to_yolo import YOLO_SKY, YOLO_WATER, semantic_from_class_masks


class YoloSegmenter:
    def __init__(self, weights: Path, imgsz: int = 1024, conf: float = 0.25, unassigned: int = OBSTACLE,
                 device: str = "cuda"):
        if unassigned not in (OBSTACLE, WATER):
            raise ValueError("unassigned must be OBSTACLE or WATER")
        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.conf = conf
        self.unassigned = unassigned
        self.device = device

    def predict_instances(self, image: np.ndarray) -> list[Instance]:
        """RGB uint8 image -> instances with masks at the original resolution."""
        # Ultralytics expects BGR for numpy input
        r = self.model.predict(image[..., ::-1], imgsz=self.imgsz, conf=self.conf, retina_masks=True,
                               device=self.device, verbose=False)[0]
        if r.masks is None:
            return []
        masks = r.masks.data.cpu().numpy() > 0.5
        classes = r.boxes.cls.cpu().numpy().astype(int)
        scores = r.boxes.conf.cpu().numpy()
        return [Instance(cls=int(c), mask=m, score=float(s)) for c, m, s in zip(classes, masks, scores)]

    def to_semantic(self, instances: list[Instance], hw: tuple[int, int]) -> np.ndarray:
        class_masks: dict[int, np.ndarray] = {}
        for inst in instances:
            m = class_masks.setdefault(inst.cls, np.zeros(hw, bool))
            m |= inst.mask
        return semantic_from_class_masks(class_masks, hw, unassigned=self.unassigned)

    def segment(self, frame: StereoFrame) -> Segmentation:
        hw = frame.left.shape[:2]
        instances = self.predict_instances(frame.left)
        labels = self.to_semantic(instances, hw)
        # confidence: best score of an instance covering the pixel, 0 where unassigned
        conf = np.zeros(hw, np.float32)
        for inst in instances:
            conf[inst.mask] = np.maximum(conf[inst.mask], inst.score)
        objects = [i for i in instances if i.cls not in (YOLO_WATER, YOLO_SKY)]
        return Segmentation(labels=labels, confidence=conf, instances=objects)
