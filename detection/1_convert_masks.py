"""
KROK 1: Konwersja masek RGB → format YOLO segmentacyjny

Uruchom: python 1_convert_masks.py

Kolory wykryte ze skanowania Twojej maski (RGB):
  Biały  (255,255,255) → woda/rzeka (najczęstszy kolor)
  Magenta (255,0,255)  → niebo
  Czerwony (255,0,0)   → łódź/sprzęt
  Zielony (0,255,0)    → roślinność/most (ciemny)
  Żółty  (255,255,0)   → most (jasny) lub inna klasa
  Czarny (0,0,0)       → krawędzie (ignorujemy)
"""

import cv2
import numpy as np
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# KLASY – dokładne kolory RGB z Twojej maski
# Kolory mają małą tolerancję (+/-15) bo JPEG/PNG może lekko je zmienić
# ══════════════════════════════════════════════════════════════════════════════
CLASSES = {
    0: {"name": "water",       "rgb": (255, 255, 255), "tol": 20},  # biały
    1: {"name": "sky",         "rgb": (255,   0, 255), "tol": 20},  # magenta
    2: {"name": "boat",        "rgb": (255,   0,   0), "tol": 25},  # czerwony
    3: {"name": "vegetation",  "rgb": (  0, 255,   0), "tol": 20},  # zielony
    4: {"name": "bridge",      "rgb": (255, 255,   0), "tol": 20},  # żółty
}

MIN_AREA    = 300     # min pikseli konturu – mniejsze ignoruj
EPSILON_F   = 0.005   # uproszczenie polygonu


def mask_to_yolo(mask_path: Path, label_path: Path):
    mask = cv2.imread(str(mask_path))
    if mask is None:
        print(f"  [BŁĄD] Nie można wczytać: {mask_path}")
        return 0

    h, w = mask.shape[:2]
    lines = []

    for class_id, cfg in CLASSES.items():
        r, g, b = cfg["rgb"]
        t = cfg["tol"]

        # BGR w OpenCV
        lo  = np.array([max(0, b-t), max(0, g-t), max(0, r-t)], np.uint8)
        hi  = np.array([min(255, b+t), min(255, g+t), min(255, r+t)], np.uint8)
        bin_mask = cv2.inRange(mask, lo, hi)

        # Morfologia – usuń szum
        k = np.ones((5,5), np.uint8)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN,  k)

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA:
                continue
            eps   = EPSILON_F * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            if len(approx) < 3:
                continue
            pts = approx.reshape(-1, 2)
            coords = " ".join(f"{x/w:.6f} {y/h:.6f}" for x, y in pts)
            lines.append(f"{class_id} {coords}")

    label_path.write_text("\n".join(lines))
    return len(lines)


def main():
    masks_dir  = Path("dataset/masks")
    labels_dir = Path("dataset/labels/train")
    labels_dir.mkdir(parents=True, exist_ok=True)

    masks = sorted(masks_dir.glob("*.png")) + sorted(masks_dir.glob("*.jpg"))
    if not masks:
        print("[!] Brak masek w dataset/masks/ – wklej tam pliki PNG")
        return

    print(f"Konwertuję {len(masks)} masek...")
    total = 0
    for mp in masks:
        lp = labels_dir / mp.with_suffix(".txt").name
        n  = mask_to_yolo(mp, lp)
        total += n
        print(f"  ✓ {mp.name} → {n} segmentów")
    print(f"\nGotowe! Łącznie {total} segmentów.")


if __name__ == "__main__":
    main()
