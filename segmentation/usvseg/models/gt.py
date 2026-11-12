"""GroundTruthSegmenter: returns dataset annotations through the Segmenter contract.

Lets the stereo/depth side develop against real labels before models are ready.
frame.frame_id must be the LaRS image name (file stem).
"""

import numpy as np

from common.interfaces import IGNORE, Instance, Segmentation, StereoFrame
from usvseg.data.lars import Lars
from usvseg.data.taxonomy import LARS_THING_CLASSES


class GroundTruthSegmenter:
    def __init__(self, dataset: Lars, with_instances: bool = True):
        self.dataset = dataset
        self.with_instances = with_instances

    def segment(self, frame: StereoFrame) -> Segmentation:
        sample = self.dataset.by_name(frame.frame_id)
        labels = sample.semantic().astype(np.uint8)
        confidence = (labels != IGNORE).astype(np.float32)
        instances = self._instances(sample) if self.with_instances and sample.panoptic_path else []
        return Segmentation(labels=labels, confidence=confidence, instances=instances)

    def _instances(self, sample) -> list[Instance]:
        ann = next(a for a in self.dataset.panoptic_annotations["annotations"]
                   if a["file_name"].rsplit(".", 1)[0] == sample.name)
        ids = sample.panoptic_ids()
        return [
            Instance(cls=seg["category_id"], mask=ids == seg["id"], score=1.0)
            for seg in ann["segments_info"]
            # crowd regions (groups of objects) stay obstacle pixels in labels, not instances
            if seg["category_id"] in LARS_THING_CLASSES and not seg.get("iscrowd", 0)
        ]
