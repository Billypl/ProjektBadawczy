"""
KROK 4: Trening YOLO

Uruchom: python 4_train.py
Wynik:   runs/obstacle_seg/weights/best.pt
"""

import yaml, torch
from pathlib import Path
from ultralytics import YOLO

# ── data.yaml ────────────────────────────────────────────────────────────────
DATA_YAML = Path("dataset/data.yaml")
DATA_YAML.parent.mkdir(parents=True, exist_ok=True)

cfg = {
    "path":  str(Path("dataset").resolve()),
    "train": "images/train",
    "val":   "images/val",
    "nc":    5,
    "names": {0:"water", 1:"sky", 2:"boat", 3:"vegetation", 4:"bridge"},
}
with open(DATA_YAML, "w") as f:
    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
print(f"✓ Zapisano {DATA_YAML}")

# ── parametry ────────────────────────────────────────────────────────────────
EPOCHS = 100
IMGSZ  = 640
BATCH  = 8          # zmniejsz do 4 jeśli mało VRAM
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"Trening na: {'GPU' if DEVICE == 0 else 'CPU'}")

# ── trening ──────────────────────────────────────────────────────────────────
model = YOLO("yolo11n-seg.pt")   # mały i szybki; zmień na yolo11m-seg.pt dla lepszej jakości

model.train(
    data     = str(DATA_YAML),
    epochs   = EPOCHS,
    imgsz    = IMGSZ,
    batch    = BATCH,
    device   = DEVICE,
    project  = "runs",
    name     = "obstacle_seg",
    flipud   = 0.2,
    fliplr   = 0.5,
    degrees  = 5.0,
    hsv_h    = 0.02,
    hsv_s    = 0.4,
    hsv_v    = 0.3,
    mosaic   = 1.0,
    patience = 30,
    save     = True,
    verbose  = True,
)
print("\n✓ Trening zakończony!")
print("  Najlepszy model: runs/obstacle_seg/weights/best.pt")
