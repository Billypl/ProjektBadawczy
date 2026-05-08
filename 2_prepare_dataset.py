"""
KROK 2: Podział na train/val (80/20) i weryfikacja par

Uruchom: python 2_prepare_dataset.py
"""

import shutil, random
from pathlib import Path

VAL_SPLIT = 0.2
SEED = 42

def main():
    img_src = Path("dataset/images/all")
    lbl_src = Path("dataset/labels/train")

    for d in ["dataset/images/train", "dataset/images/val",
              "dataset/labels/val"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    images = sorted(img_src.glob("*.png")) + sorted(img_src.glob("*.jpg"))
    pairs, missing = [], []
    for img in images:
        lbl = lbl_src / img.with_suffix(".txt").name
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            missing.append(img.name)

    if missing:
        print(f"[!] Brak etykiet dla {len(missing)} obrazów: {missing[:5]}...")
    if not pairs:
        print("[!] Brak par obraz+etykieta!")
        return

    random.seed(SEED)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_SPLIT))
    val_pairs   = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_img = Path("dataset/images/train")
    val_img   = Path("dataset/images/val")
    train_lbl = Path("dataset/labels/train")
    val_lbl   = Path("dataset/labels/val")

    for img, lbl in val_pairs:
        shutil.copy(img, val_img / img.name)
        shutil.copy(lbl, val_lbl / lbl.name)

    # train images są już w dataset/images/all – skopiuj do train
    for img, lbl in train_pairs:
        shutil.copy(img, train_img / img.name)
        # etykiety train już są na miejscu

    print(f"Łącznie: {len(pairs)} par")
    print(f"  Train: {len(train_pairs)}")
    print(f"  Val:   {len(val_pairs)}")

if __name__ == "__main__":
    main()
