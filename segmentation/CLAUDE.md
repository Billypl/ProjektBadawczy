# CLAUDE.md - USV: segmentacja wody i przeszkód (część Billy'ego)

Ten plik to kontekst i instrukcje dla Claude Code. Czytaj go na początku każdej sesji.

## Jak ze mną pracować

- Odpowiadaj po polsku. Kod, komentarze, nazwy plików, logi i opisy commitów po angielsku (raport końcowy będzie po angielsku).
- Nie używaj długich myślników, tylko zwykłego "-".
- Moje tło: C#/.NET i OOP. Decyzje ML tłumacz krótko i konkretnie: co, dlaczego, jaki koszt.
- Przed czymś, co zajmie >2 h obliczeń albo pobierze >5 GB, zapytaj mnie.
- Gdy utkniesz na tym samym problemie ze środowiskiem po 2 próbach, zatrzymaj się i opisz problem zamiast kombinować dalej.

## Projekt

- Zespołowy Projekt Badawczy (ZPB) na Politechnice Gdańskiej, zespół 3 osób: Billy, Filip, Hania.
- Temat: wykrywanie przeszkód i celów oraz nawigacja USV (drona wodnego) po rzekach na podstawie video z pary kamer (stereo).
- Mamy drona wodnego i dwie kamery. Obecny etap: offline, na publicznych zbiorach danych.
- Podział pracy:
  - **Billy (ten folder, `segmentation/`):** dobór zbiorów do segmentacji + model segmentacji + ewaluacja.
  - **Filip i Hania:** zbiory stereo, dysparycja, przeliczenie px -> metry. Nie implementuj ich części, najwyżej zaślepkę.
  - **Na razie niczyje:** fuzja segmentacji z głębią (wolna przestrzeń), planer, symulator, własne nagrania.
- Terminy ZPB: koniec prac 31.12.2026, raport jako artykuł po angielsku w szablonie IEEE, prezentacje w styczniu 2027.
- **Zakres obecnych sesji: kod, research, trening, ewaluacja. Nie piszemy artykułu.** Zbieraj za to materiał do niego (tabele, wykresy, notatki w `results/notes.md`).
- Folder `../detection/` w repo to wcześniejsza praca (YOLO na innym zbiorze). Nie ruszaj go.

## Decyzje już podjęte (nie zmieniaj bez pytania)

1. **Rdzeń: segmentacja semantyczna** na klasy przeszkoda / woda / niebo + ignoruj. To standard w literaturze USV i format benchmarku LaRS. Granica woda-przeszkoda w każdej kolumnie obrazu to wejście do późniejszej mapy wolnej przestrzeni.
2. **YOLO26-seg (Ultralytics)** jako metoda porównawcza (segmentacja instancyjna), później jako detektor obiektów ruchomych do śledzenia.
3. **Baseline bez treningu: eWaSR** z gotowymi wagami.
4. **Metryki liczy oficjalny `lars_evaluator`.** Nie implementuj własnych wersji metryk do raportowania.
5. Duże modele panoptyczne (Mask2Former + DINOv3) traktujemy tylko jako odniesienie z literatury, nie trenujemy ich.
6. SAM 3 dopiero później, do półautomatycznego etykietowania własnych nagrań z rzeki.

## Hipotezy (po co są eksperymenty)

- **H1 (główna dla tego folderu):** segmentacja semantyczna daje wyższe F1 przeszkód i dokładniejszą krawędź wody niż segmentacja instancyjna YOLO przy porównywalnym FPS.
- H2 (później, wspólnie z Filipem i Hanią): odrzucanie pikseli "przeszkód", które w stereo leżą pod płaszczyzną wody, zmniejsza liczbę fałszywych alarmów od odbić.
- H3 (później): model trenowany głównie na morzu traci skuteczność na rzekach. Test na rzecznych scenach LaRS i własnych nagraniach.

## Klasy i konwencje (kontrakt z zespołem)

ID klas zgodne z LaRS (`semantic_masks`):

```python
OBSTACLE, WATER, SKY, IGNORE = 0, 1, 2, 255
```

Uwaga: we wcześniejszym szkicu interfejsu było WATER=0, SKY=1, OBSTACLE=2. Zmienione, żeby zgadzało się z LaRS i MaSTr. Interfejs nie jest jeszcze ustalony z Filipem i Hanią, można go zmieniać w razie potrzeby.

- Maski predykcji muszą mieć **rozdzielczość oryginalnego obrazu** (tego wymaga evaluator).
- Przy integracji z zespołem: wszystko w geometrii **lewego obrazu po rektyfikacji**.
- Etykiety zapisuj jako PNG uint8, nigdy JPEG. Maski skaluj zawsze metodą nearest.
- Układ kamery jak w OpenCV (x w prawo, y w dół, z do przodu), jednostki w metrach.

Wspólny kontrakt: `common/interfaces.py` (`StereoFrame`, `Instance`, `Segmentation`, `Segmenter`).
`usvseg/models/gt.py` zawiera `GroundTruthSegmenter`, który zwraca adnotacje ze zbioru; Filip i Hania mogą go używać, zanim modele będą gotowe.

## Dane

Zbiory pobiera Billy ręcznie (LaRS wymaga formularza). **Nie obchodź formularzy rejestracyjnych.** Surowe dane trafiają do `data/raw/<nazwa>/`, nic z `data/` nie trafia do gita.

| Zbiór | Rola | Uwagi |
|---|---|---|
| LaRS v1.0.0 | trening + walidacja | jeziora, rzeki, morza; maski semantyczne i panoptyczne |
| MaSTr1325 (+ MaSTr1478) | dodatkowy trening | morze przybrzeżne; MaSTr1478 dodaje trudne sceny z odbiciami (repo WaSR-T) |
| USVInland | później, test na rzece | etykiety tylko wody, więc tylko metryki wody |

**LaRS**
- Pliki: `lars_v1.0.0_images.zip` (~1 GB) i `lars_v1.0.0_annotations.zip` (~22 MB). Sekwencji (`lars_v1.0.0_images_seq.zip`, 13.6 GB) na razie **nie** pobieramy.
- Adnotacje: `semantic_masks/` (PNG: 0 przeszkody, 1 woda, 2 niebo, 255 ignoruj), `panoptic_masks/` + `panoptic_annotations.json` (format COCO panoptic), `image_annotations.json` (atrybuty scen).
- Klasy panoptyczne: 1 Static Obstacle, 3 Water, 5 Sky (tło) oraz obiekty: 11 Boat/ship, 12 Row boats, 13 Paddle board, 14 Buoy, 15 Swimmer, 16 Animal, 17 Float, 19 Other.
- Podział: train ok. 2605, val ok. 198, test ok. 1203. Etykiety testu są ukryte (serwer ewaluacyjny).
- **Split `dev` = 10% train, seed 42 (`usvseg/data/lars.py`), do strojenia.** Val służy tylko do raportowania, żeby nie stroić na tym, co raportujemy.
- Przed napisaniem mapowań **sprawdź rzeczywiste pliki**: `np.unique` na maskach, `categories` w JSON, rozdzielczości. Liczby powyżej pochodzą z dokumentacji, pliki mają pierwszeństwo.

**MaSTr1325**: wartości masek 0 przeszkody, 1 woda, 2 niebo, **4 ignoruj -> mapuj na 255**.

## Modele

| ID | Model | Źródło | Licencja | Uwagi |
|---|---|---|---|---|
| M0 | eWaSR `ewasr_resnet18` | github.com/tersekmatija/eWaSR | Apache-2.0 | bez treningu; wariant **bez IMU** (LaRS nie ma IMU); wagi trenowane na MaSTr1325 (512x384); osobne venv `third_party/eWaSR/.venv` |
| M1 | SegFormer-B2 | HF `transformers`, encoder `nvidia/mit-b2` | do sprawdzenia przy wagach | `num_labels=3`, `ignore_index=255` |
| M2 | YOLO26-seg | `ultralytics`, start `yolo26s-seg.pt` (`n` do testów szybkości) | AGPL-3.0 | kod Ultralytics trzymaj w osobnym module `usvseg/models/yolo_seg.py` |
| M3 | PIDNet-S (stretch) | github.com/XuJiacong/PIDNet | do sprawdzenia | dobrze znosi kwantyzację INT8, ważne dla sprzętu pokładowego |

**SegFormer, trening**
- Najpierw HF `transformers` z własną pętlą albo `Trainer`. Starter kit `lojzezust/mmsegmentation-macvi` (MMSeg 0.x) traktuj jako źródło hiperparametrów (rozmiary cropów, harmonogram). Instaluj go tylko, jeśli HF zawiedzie.
- Augmentacje: flip **tylko poziomy**, zmiana skali + crop, jitter kolorów, rotacja najwyżej ±5-10° (przechył łodzi). **Bez flipu pionowego**, bo niebo jest zawsze u góry i model ma z tego korzystać.
- Pierwszy pełny trening krótki (rząd 20-40 epok). Dłuższy dopiero, gdy pipeline działa.

**YOLO26-seg, konwersja danych**
- Źródło: LaRS panoptic (PNG + JSON). ID segmentu w PNG: `id = R + 256*G + 256**2*B` (standard COCO panoptic). Zweryfikuj to na danych.
- Klasy YOLO (11): Static Obstacle, Water, Sky jako "instancje" (każdy spójny region osobno) + 8 klas obiektów.
- Wielokąty z `cv2.findContours(..., cv2.RETR_CCOMP, ...)`. **Format YOLO nie ma dziur.** Dla każdego obrazu zmierz, jaki % pikseli zmienia klasę po konwersji (wielokąty -> maska) i zapisz w statystykach. Jeśli strata jest duża, zastosuj cięcie typu "keyhole" (dziura połączona z obrysem cienkim cięciem).
- Upraszczanie konturów nie może gubić małych obiektów. Pilnuj minimalnego pola i liczby wierzchołków.
- `ultralytics.data.converter.convert_segment_masks_to_yolo_seg` istnieje, ale zakłada inne kodowanie masek. Możesz go użyć tylko po sprawdzeniu, że daje poprawny wynik.
- **Predykcje YOLO -> mapa semantyczna:** obiekty i Static Obstacle -> OBSTACLE, potem SKY, potem WATER. Piksele bez przypisania domyślnie -> OBSTACLE (wariant ostrożny). Zrób to parametrem i zaraportuj oba warianty.

## Ewaluacja

- Narzędzie: `github.com/lojzezust/lars_evaluator` w `third_party/` (zależności lekkie, działa w głównym venv; dodatkowo `yacs`).
- Wrapper: `usvseg/eval/run_lars_eval.py` generuje config, uruchamia evaluator, dopisuje wiersz do `results/results.csv`.
- Zapisujemy **kolorowe PNG RGB**: niebo `[90, 75, 164]`, woda `[41, 167, 224]`, przeszkoda `[247, 195, 37]` (serwer online przyjmuje tylko tę wersję).
- Wyniki evaluatora: `summary.json` (całość, klucze `F1`, `WE_acc` = μ, `mIoU`, `Q`), `frames.csv` (per klatka), `segments.csv` (TP/FP/FN). `frames.csv` + `image_annotations.json` pozwalają policzyć wyniki dla podzbiorów, np. rzeki i odbicia.
- Metryki: **F1** (przeszkody dynamiczne), **μ** (dokładność krawędzi wody, przeszkody statyczne), mIoU, **Q = F1 · mIoU**. Główne to F1 i μ. Samo mIoU wprowadza w błąd, bo woda i niebo zajmują większość obrazu.
- **FPS:** batch 1, stała rozdzielczość wejścia, 50 iteracji rozgrzewki, 200 pomiarowych, `torch.cuda.synchronize()`, czas łącznie z pre- i postprocessingiem (dla YOLO z budowaniem mapy semantycznej). Zapisuj GPU i rozdzielczość.
- Liczby do raportu bierz tylko z wyjścia evaluatora. Nie przepisuj ich ręcznie, skrypt dopisuje je do `results/results.csv`.

**Punkty odniesienia (benchmark LaRS, zbiór testowy; na val wyjdą inne):**

| Metoda | F1 | μ | mIoU | Q |
|---|---|---|---|---|
| KNet (Swin-T) | 73.4 | 78.8 | 97.2 | 71.3 |
| SegFormer (MiT-B2) | 70.0 | 78.6 | 96.8 | 67.8 |
| DeepLabv3 (ResNet-101) | 66.1 | 77.5 | 95.2 | 62.9 |
| WaSR (ResNet-101) | 61.6 | 71.0 | 96.6 | 59.5 |

Kontekst sprzętowy: zwycięzca toru embedded MaCVi 2026 (DSOS-Net, OAK4, INT8) miał Q 61.9. Tam SegFormer i SegNeXt traciły 67-82% mIoU po kwantyzacji INT8.

## Struktura folderu

```
segmentation/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .gitignore              # .venv/, data/, runs/, third_party/, *.pt, *.pth, *.onnx
├── configs/                # one YAML per experiment
├── common/
│   └── interfaces.py       # shared contract with Filip and Hania
├── usvseg/
│   ├── data/
│   │   ├── taxonomy.py     # class constants, colors, mappings
│   │   ├── lars.py         # LaRS loader + dev split
│   │   ├── mastr.py        # MaSTr loader (ignore 4 -> 255)
│   │   └── to_yolo.py      # LaRS panoptic -> YOLO polygons + loss stats
│   ├── models/
│   │   ├── ewasr.py        # wrapper around third_party/eWaSR
│   │   ├── segformer.py
│   │   ├── yolo_seg.py     # Ultralytics (AGPL) isolated here
│   │   └── gt.py           # GroundTruthSegmenter
│   ├── train/
│   ├── eval/
│   │   ├── export_lars.py  # write predictions in evaluator format
│   │   ├── run_lars_eval.py
│   │   └── fps.py
│   └── viz/
│       └── overlay.py      # mask overlays, side-by-side comparisons
├── scripts/                # thin CLI entry points
├── tests/                  # taxonomy mapping, color coding, YOLO round-trip
├── third_party/            # cloned repos (gitignored)
├── data/                   # gitignored
├── runs/                   # training outputs (gitignored)
└── results/
    ├── results.csv         # one row per evaluation
    ├── figures/
    └── notes.md            # experiment log: what, why, result, conclusion
```

Kolumny `results.csv`: `date, git_commit, exp_name, model, train_data, eval_split, input_res, F1, mu, mIoU, Q, fps, hardware, notes`.

## Plan pierwszej sesji (po kolei)

**Krok 0 - środowisko** (zrobione 2026-11-08)
- RTX 5070 Ti, torch 2.8.0+cu128, `python -m venv` (brak `uv`), wersje przypięte w `pyproject.toml`.

**Krok 1 - dane LaRS**
- Loader, wydzielenie `dev`, statystyki (liczba obrazów, rozdzielczości, rozkład klas, unikalne wartości masek): `scripts/lars_stats.py`.
- Wygeneruj 20 losowych nakładek maska + obraz do `runs/checks/lars/`.
- Gotowe, gdy: statystyki są w `results/notes.md` i Billy rzucił okiem na nakładki.

**Krok 2 - baseline eWaSR (M0)**
- Uruchom `ewasr_resnet18` (bez IMU) na LaRS val, zapisz predykcje w formacie evaluatora, puść `lars_evaluator`.
- Gotowe, gdy: pierwszy wiersz w `results.csv`.

**Krok 3 - SegFormer-B2 (M1)**
- Smoke run (20 obrazów, 1 epoka), potem pełny trening w tle.
- Ewaluacja na val.
- Gotowe, gdy: wiersz w `results.csv` + checkpoint w `runs/`.

**Krok 4 - YOLO26-seg (M2)** (konwersję możesz robić, gdy trenuje SegFormer)
- Konwersja LaRS -> YOLO ze statystyką strat i nakładkami do `runs/checks/yolo/`. Test: konwersja tam i z powrotem.
- Smoke run, potem pełny trening `yolo26s-seg.pt`.
- Predykcje -> mapa semantyczna (oba warianty pikseli bez przypisania) -> evaluator.
- Gotowe, gdy: 2 wiersze w `results.csv`.

**Krok 5 - podsumowanie**
- Tabela porównawcza M0-M2, 10 obrazów porównawczych (obraz | GT | M1 | M2) do `results/figures/`.
- Wnioski i obserwowane błędy (małe obiekty, odbicia, krawędź wody) w `notes.md`.

**Stretch (jeśli zostanie czas):** FPS dla wszystkich modeli; wyniki dla rzecznego podzbioru LaRS; trening na LaRS + MaSTr1478; PIDNet-S.

## Zasady pracy

- **Najpierw zajrzyj w dane, potem pisz mapowania.** Żadnych założonych ID klas bez sprawdzenia na plikach.
- Po każdej konwersji danych: nakładki do wglądu + test jednostkowy round-trip.
- Każdy skrypt treningowy najpierw w trybie smoke (mały podzbiór, 1 epoka).
- Długie treningi w tle (`nohup` albo `tmux`), log do `runs/<exp>/train.log`. Podaj Billy'emu komendę do podglądu i nie blokuj sesji czekaniem.
- Stały seed. Config każdego eksperymentu zapisuj obok wyników w `runs/<exp>/`.
- Do gita nie trafiają: dane, wagi, checkpointy, pliki >10 MB.
- Commituj małe, działające kroki z opisem po angielsku.
- Nie ruszaj kodu stereo i głębi (część Filipa i Hani).
- Każdy eksperyment opisz w `results/notes.md` w formacie: co, dlaczego, wynik, wniosek. Z tego powstanie potem sekcja artykułu.

## Znane pułapki

- Ignorowanie: MaSTr używa wartości 4, LaRS 255. Ujednolicaj do 255.
- LaRS ma obrazy w wielu rozdzielczościach. Model może pracować w stałej, ale predykcję przeskaluj (nearest) do oryginału przed zapisem.
- Format YOLO: brak dziur w wielokątach, zgrubne maski, woda i niebo jako sztuczne "instancje". Wszystko mierz, niczego nie zakładaj.
- Małe i odległe obiekty mają po kilka pikseli. Pilnuj ich przy konwersji, cropach i skalowaniu w dół.
- Odbicia w wodzie to główne źródło fałszywych przeszkód. Zbieraj przykłady do `notes.md`.
- Repozytoria badawcze (eWaSR, evaluator) mogą mieć stare zależności. Wtedy osobne venv w `third_party/`, nie psuj głównego środowiska.
- eWaSR bez IMU: `WaSR.forward` i tak czyta `x['imu_mask']`, wrapper podaje zera (decoder je ignoruje przy `imu=False`).
- Licencje zbiorów są niekomercyjne lub do sprawdzenia. Nie redystrybuuj danych.

## Odpowiedzi na otwarte pytania

- GPU: lokalny RTX 5070 Ti (16 GB), WSL2.
- Sprzęt na łodzi: na razie najważniejszy jest proof of concept, więc PIDNet-S i kwantyzacja nie są priorytetem.
- Interfejs z Filipem i Hanią nie jest ustalony, można go zmieniać w razie potrzeby.
