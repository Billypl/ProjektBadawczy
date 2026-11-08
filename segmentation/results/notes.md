# Experiment log

Format per entry: what, why, result, conclusion.

## 2026-11-08 - Environment

- What: main venv (`.venv`, Python 3.12, torch 2.8.0+cu128) and a separate eWaSR venv (`third_party/eWaSR/.venv`, timm 0.6.5).
- Why: RTX 5070 Ti (Blackwell, sm_120) needs CUDA 12.8 wheels; eWaSR pins an old timm.
- Result: `torch.cuda.is_available() == True` in both venvs.
- Note: eWaSR no-IMU model still reads `x['imu_mask']` in `WaSR.forward`; the wrapper passes zeros (ignored by the decoder when `imu=False`).
- Note: eWaSR was trained on MaSTr1325 at 512x384, so M0 resizes LaRS input to 512x384 and upsamples logits (bilinear) back to the original size.

## LaRS dataset statistics

Source: `scripts/lars_stats.py`, dev = 10% of train (seed 42).

### train (2345 images, 0 without semantic mask)

Resolutions (W x H: count): 1278x958: 1172, 1920x1080: 453, 1280x720: 415, 2208x1242: 132, 640x480: 72, 1280x1024: 46, 640x400: 46, 1920x1440: 3, 1280x702: 3, 1280x676: 3

| mask value | class | pixel share |
|---|---|---|
| 0 | obstacle | 19.68% |
| 1 | water | 44.18% |
| 2 | sky | 35.89% |
| 255 | ignore | 0.25% |

### dev (260 images, 0 without semantic mask)

Resolutions (W x H: count): 1278x958: 151, 1920x1080: 56, 1280x720: 31, 2208x1242: 8, 1280x1024: 8, 640x480: 3, 640x400: 3

| mask value | class | pixel share |
|---|---|---|
| 0 | obstacle | 16.03% |
| 1 | water | 45.56% |
| 2 | sky | 38.30% |
| 255 | ignore | 0.12% |

### val (198 images, 0 without semantic mask)

Resolutions (W x H: count): 1920x1080: 79, 1280x720: 71, 2208x1242: 15, 640x400: 11, 1280x1024: 10, 640x480: 7, 1280x676: 4, 1920x1440: 1

| mask value | class | pixel share |
|---|---|---|
| 0 | obstacle | 23.82% |
| 1 | water | 49.33% |
| 2 | sky | 25.44% |
| 255 | ignore | 1.41% |

### test (1203 images, 1203 without semantic mask)

Resolutions (W x H: count): 1920x1080: 441, 1280x720: 426, 2208x1242: 137, 640x400: 83, 1280x1024: 58, 640x480: 29, 1280x676: 13, 1280x702: 7, 720x576: 3, 1920x1440: 3, 1440x1080: 2, 2688x1520: 1

### panoptic categories (train)

| id | name | isthing | segments |
|---|---|---|---|
| 1 | Static Obstacle | 0 | 2533 |
| 3 | Water | 0 | 2605 |
| 5 | Sky | 0 | 2492 |
| 11 | Boat/ship | 1 | 5921 |
| 12 | Row boats | 1 | 444 |
| 13 | Paddle board | 1 | 154 |
| 14 | Buoy | 1 | 1568 |
| 15 | Swimmer | 1 | 349 |
| 16 | Animal | 1 | 361 |
| 17 | Float | 1 | 23 |
| 19 | Other | 1 | 501 |

Checks on the actual files (2026-11-08):
- Semantic masks: uint8, values exactly {0, 1, 2, 255}, same size as the image.
- Panoptic PNG: R = category id, G/B = instance index. The COCO formula `R + 256*G + 256^2*B` reproduces the JSON segment ids for every train/val image (0 mismatches); id 0 = unlabelled.
- Crowd segments (`iscrowd=1`, groups of objects): 184 in train (162 Boat/ship), 25 in val. They are obstacle pixels in the semantic masks; GroundTruthSegmenter does not return them as instances.
- Scene types: train 568 river / 2036 sea, val 79 river / 119 sea, test 487 river / 716 sea. Reflections in val: 84 none, 103 moderate, 10 heavy.
- **LaRS train already contains MaSTr1325** (1323 `mastr_*` images, all 1278x958). Val and test contain no MaSTr images. Consequences: (1) "LaRS + MaSTr1325" training would duplicate data, only MaSTr1478 extras add anything; (2) eWaSR weights (trained on MaSTr1325) saw about half of LaRS train, but nothing from val, so the M0 val result is not leaked.
- Resolution mix differs: half of train is 1278x958 (MaSTr), val/test are mostly 1920x1080 and 1280x720.
