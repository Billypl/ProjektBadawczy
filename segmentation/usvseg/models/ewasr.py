"""eWaSR (M0) wrapper around third_party/eWaSR.

Runs only in the eWaSR venv (third_party/eWaSR/.venv) because of its pinned
timm/pytorch-lightning. Uses the no-IMU variant; WaSR.forward still reads
x['imu_mask'] unconditionally, so a zero mask is passed. The EWaSR decoder
only uses it when imu=True, so it has no effect on the output.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.interfaces import Segmentation, StereoFrame

EWASR_DIR = Path(__file__).resolve().parents[2] / "third_party" / "eWaSR"
DEFAULT_WEIGHTS = EWASR_DIR / "pretrained" / "ewasr_resnet18.pth"

# torchvision/PyTorch Hub normalisation, as in eWaSR's PytorchHubNormalization.
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class EwasrSegmenter:
    def __init__(self, weights: Path = DEFAULT_WEIGHTS, input_hw: tuple[int, int] | None = (384, 512),
                 architecture: str = "ewasr_resnet18", device: str = "cuda"):
        if str(EWASR_DIR) not in sys.path:
            sys.path.insert(0, str(EWASR_DIR))
        import wasr.models as models
        from wasr.utils import load_weights

        self.input_hw = input_hw
        self.device = torch.device(device)
        model = models.get_model(architecture, num_classes=3, pretrained=False)
        model.load_state_dict(load_weights(str(weights)))
        self.model = model.eval().to(self.device)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """RGB uint8 HxWx3 -> (labels HxW uint8, confidence HxW float32) at input resolution."""
        h, w = image.shape[:2]
        x = (image.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device)
        if self.input_hw is not None:
            x = F.interpolate(x, size=self.input_hw, mode="bilinear", align_corners=False)
        imu = torch.zeros(1, *x.shape[2:], device=self.device)
        logits = self.model({"image": x, "imu_mask": imu})["out"]
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        prob = logits.softmax(1)[0]
        conf, labels = prob.max(0)
        return labels.byte().cpu().numpy(), conf.float().cpu().numpy()

    def segment(self, frame: StereoFrame) -> Segmentation:
        labels, conf = self.predict(frame.left)
        return Segmentation(labels=labels, confidence=conf)
