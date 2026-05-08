"""
KROK 3: Weryfikacja – nakłada polygony na oryginalne zdjęcia

Uruchom: python 3_verify.py
Sprawdź: verify_*.jpg – czy pokrywają się z maską?
"""

import cv2, numpy as np
from pathlib import Path

# Kolory do rysowania (BGR)
COLORS = {
    0: (255, 255, 255),  # water     – biały
    1: (255,   0, 255),  # sky       – magenta
    2: (  0,   0, 255),  # boat      – czerwony (w BGR)
    3: (  0, 255,   0),  # veg       – zielony
    4: (  0, 255, 255),  # bridge    – żółty (w BGR)
}
NAMES = {0:"water", 1:"sky", 2:"boat", 3:"vegetation", 4:"bridge"}

def draw(img_path, lbl_path, out_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [!] Nie mogę wczytać: {img_path}"); return
    h, w = img.shape[:2]
    overlay = img.copy()

    if not lbl_path.exists():
        print(f"  [!] Brak etykiety: {lbl_path}"); return

    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if not parts: continue
        cid   = int(parts[0])
        coords = list(map(float, parts[1:]))
        pts = np.array([[int(coords[i]*w), int(coords[i+1]*h)]
                        for i in range(0, len(coords), 2)], np.int32)
        color = COLORS.get(cid, (128,128,128))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 2)
        if len(pts):
            cv2.putText(img, NAMES.get(cid, str(cid)), tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    result = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    cv2.imwrite(str(out_path), result)
    print(f"  ✓ {out_path}")

def main():
    for img_dir in [Path("dataset/images/train"), Path("dataset/images/all")]:
        imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
        if imgs:
            break
    lbl_dir = Path("dataset/labels/train")

    if not imgs:
        print("[!] Brak zdjęć – uruchom najpierw 1_convert_masks.py i 2_prepare_dataset.py")
        return

    print(f"Weryfikuję pierwsze {min(3, len(imgs))} zdjęcia...")
    for img_path in imgs[:3]:
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        draw(img_path, lbl_path, Path(f"verify_{img_path.stem}.jpg"))

if __name__ == "__main__":
    main()
