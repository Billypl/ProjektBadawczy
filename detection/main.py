from ultralytics import YOLO
import shutil
from pathlib import Path

# 4. Policz mAP
model = YOLO("runs/segment/runs/obstacle_seg/weights/best.pt")

metrics = model.val(
    data="dataset/data.yaml",
    split="test",
)

print(f"\nmAP50:     {metrics.seg.map50:.3f}")
print(f"mAP50-95:  {metrics.seg.map:.3f}")
print(f"Precision: {metrics.seg.mp:.3f}")
print(f"Recall:    {metrics.seg.mr:.3f}")

print("\nPer klasa:")
for i, name in model.names.items():
    print(f"  {name:12s}  mAP50={metrics.seg.maps[i]:.3f}")