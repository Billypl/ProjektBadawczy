# YOLO Segmentacja – Instrukcja (wersja 2 – maska RGB)

## Dataset
full: https://purr.purdue.edu/publications/4072/citations/1  
smaller: https://drive.google.com/drive/folders/16lKo7SE3B7H5wQqv3PcfhUvOS-yievhU


## Struktura folderów – CO GDZIE WKLEIĆ

```
yolo_project/
│
├── dataset/
│   ├── masks/            ← WKLEJ TUTAJ kolorowe maski RGB (np. 343.png)
│   └── images/
│       └── all/          ← WKLEJ TUTAJ oryginalne zdjęcia (np. 343.png)
│                            NAZWA PLIKU MUSI BYĆ IDENTYCZNA z maską!
│
├── 1_convert_masks.py
├── 2_prepare_dataset.py
├── 3_verify.py
├── 4_train.py
├── 5_test.py
└── requirements.txt
```

---

## Kolory klas w Twojej masce RGB

| ID | Nazwa       | Kolor RGB        | Kolor wizualny |
|----|-------------|------------------|----------------|
| 0  | water       | (255, 255, 255)  | ⬜ Biały        |
| 1  | sky         | (255,   0, 255)  | 🟣 Magenta      |
| 2  | boat        | (255,   0,   0)  | 🔴 Czerwony     |
| 3  | vegetation  | (  0, 255,   0)  | 🟢 Zielony      |
| 4  | bridge      | (255, 255,   0)  | 🟡 Żółty        |

---

## Krok po kroku

### Krok 0 – Instalacja
```bash
pip install -r requirements.txt
```

### Krok 1 – Konwersja masek → etykiety YOLO
```bash
python 1_convert_masks.py
```
Czyta pliki z `dataset/masks/`, zapisuje `.txt` do `dataset/labels/train/`

### Krok 2 – Podział train/val
```bash
python 2_prepare_dataset.py
```
Automatyczny podział 80/20.

### Krok 3 – Weryfikacja (OBOWIĄZKOWE!)
```bash
python 3_verify.py
```
Otwórz pliki `verify_*.jpg`.  
**Polygony muszą pokrywać się z obszarami na oryginalnym zdjęciu.**  
Jeśli nie – edytuj kolory w `1_convert_masks.py` (sekcja `CLASSES`).

### Krok 4 – Trening
```bash
python 4_train.py
```
- GPU: ~20–40 min dla 100 epok  
- CPU: kilka godzin  
- Wynik: `runs/obstacle_seg/weights/best.pt`

### Krok 5 – Test
```bash
python 5_test.py --image twoje_zdjecie.png
python 5_test.py --folder folder_ze_zdjeciami/
```

---

## Typowe problemy

| Problem | Rozwiązanie |
|---------|-------------|
| `verify_*.jpg` – błędne kolory | Sprawdź RGB masek pipetą w GIMP/Photoshop |
| Model nie wykrywa | Obniż `CONF` w `5_test.py` do 0.2 |
| Mało danych (<30 obrazów) | Zbierz więcej lub użyj augmentacji |
| CUDA out of memory | Zmniejsz `BATCH` do 4 lub 2 |
| CPU bardzo wolny | Zmień model na `yolo11n-seg.pt` (już jest) |

---

## Jak sprawdzić dokładny kolor w masce

```python
import cv2, numpy as np
mask = cv2.imread("dataset/masks/twoja_maska.png")
# Wyświetl top kolory:
pixels = mask.reshape(-1, 3)
unique, counts = np.unique(pixels, axis=0, return_counts=True)
for i in np.argsort(-counts)[:10]:
    b,g,r = unique[i]
    print(f"RGB=({r},{g},{b})  count={counts[i]}")
```
