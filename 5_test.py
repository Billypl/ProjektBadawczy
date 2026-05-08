"""
KROK 5: Testowanie wytrenowanego modelu

Uruchom: python 5_test.py --image foto.png
         python 5_test.py --folder moje_zdjecia/
         python 5_test.py          (testuje zbiór val)
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

WEIGHTS    = "runs/segment/runs/obstacle_seg/weights/best.pt"
CONF       = 0.35
IOU        = 0.45
CLASS_NAMES = {0:"water", 1:"sky", 2:"boat", 3:"bridge", 4:"vegetation"}

def run(source, out_dir):
    model = YOLO(WEIGHTS)

    model.model.names = {
        0: "water",
        1: "sky",
        2: "boat",
        3: "bridge",  # ← zamienione
        4: "vegetation",  # ← zamienione
    }

    results = model(source, conf=CONF, iou=IOU, imgsz=640,
                    save=True, project=str(out_dir), name="results",
                    retina_masks=True)
    for r in results:
        if r.masks is None:
            print(f"  {Path(r.path).name}: brak detekcji")
        else:
            cls  = [CLASS_NAMES.get(int(c), str(c)) for c in r.boxes.cls]
            conf = [f"{c:.2f}" for c in r.boxes.conf]
            print(f"  {Path(r.path).name}: {list(zip(cls, conf))}")
    print(f"\n✓ Wyniki: {out_dir}/results/")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image",  type=str)
    p.add_argument("--folder", type=str)
    p.add_argument("--out",    type=str, default="output")
    args = p.parse_args()

    if not Path(WEIGHTS).exists():
        print(f"[!] Nie znaleziono modelu: {WEIGHTS}")
        print("    Uruchom najpierw 4_train.py")
        return

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if   args.image:  run(args.image, out)
    elif args.folder: run(args.folder, out)
    else:             run("dataset/images/val", out)

if __name__ == "__main__":
    main()
