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

## 2026-11-08 - M0: eWaSR baseline on LaRS val

- What: `ewasr_resnet18` (no IMU), official weights trained on MaSTr1325, no training on LaRS. Input resized to 512x384 (MaSTr training resolution), logits upsampled bilinearly to the original size. Run: `runs/m0_ewasr_val/`.
- Why: zero-cost baseline; shows how a coastal-sea model transfers to the more diverse LaRS (lakes, rivers, other sensors).
- Result (lars_evaluator, val, 198 images): **F1 49.3, mu 59.3, mIoU 85.1, Q 42.0**. Detection: TP 837, FP 1450, FN 270 -> precision 36.6, recall 75.6. IoU obstacle 76.2, water 88.7, sky 90.3.
- Breakdown (`python -m usvseg.eval.subsets --exp m0_ewasr_val`, aggregates evaluator TP/FP/FN per frame):

| subset | n | TP | FP | FN | Pr | Re | F1 |
|---|---|---|---|---|---|---|---|
| river_like | 79 | 128 | 777 | 21 | 14.1 | 85.9 | 24.3 |
| sea_like | 119 | 709 | 673 | 249 | 51.3 | 74.0 | 60.6 |
| reflections: none | 84 | 476 | 602 | 127 | 44.2 | 78.9 | 56.6 |
| reflections: moderate | 103 | 358 | 767 | 143 | 31.8 | 71.5 | 44.0 |
| reflections: heavy | 10 | 3 | 80 | 0 | 3.6 | 100 | 7.0 |

- Observed errors (`runs/checks/m0_worst_fp/`, top-6 frames by FP):
  - Sun glitter on open sea predicted as obstacle blobs around the horizon (`inhouse_seq_155_00012`, 32 FP).
  - Narrow jungle river with brown water (`yt055_02_00114`): water predicted as sky/obstacle, foliage partly as water. Pure domain gap, MaSTr has nothing like it.
- Conclusion: the low F1 comes from false positives, not misses (recall 75.6). River scenes are much worse than sea (F1 24 vs 61), and F1 drops with reflection strength. This supports H3 (sea-trained model degrades on rivers) and motivates H2 (reflection/glitter FPs). Note the val set is small (79 river frames, 10 heavy-reflection frames), so subset numbers are noisy.
- Caveat: `mu_frames` in `subsets.py` output is a mean of per-frame WE_acc and is not the global mu; only the global numbers go to the report.

## 2026-11-08 - LaRS -> YOLO-seg conversion (for M2)

- What: `scripts/convert_lars_yolo.py`. 11 YOLO classes: static_obstacle, water, sky as pseudo-instances (one per connected component) + 8 thing classes (one polygon per instance component; crowd segments kept). Output `data/yolo/lars/` (train / dev as YOLO val / LaRS val as test).
- Why: YOLO format has no holes and no "stuff" classes; the loss of this conversion has to be known before comparing YOLO to semantic models (H1).
- Ablation (150 random train images; % of non-ignore pixels whose semantic class changes after polygons -> mask, and per-thing IoU raster vs GT):

| keyhole | eps (px) | changed mean | changed max | thing IoU mean | thing IoU min | things IoU<0.8 |
|---|---|---|---|---|---|---|
| no | 0 | 0.244% | 8.94% | 0.985 | 0.492 | 15/530 |
| no | 1 | 0.304% | 8.98% | 0.966 | 0.488 | 16/530 |
| yes | 0 | 0.000% | 0.00% | 1.000 | 0.999 | 0/530 |
| yes | 1 | 0.066% | 0.30% | 0.979 | 0.838 | 0/530 |
| yes | 2 | 0.136% | 0.60% | 0.956 | 0.737 | 5/530 |

- Choice: keyhole + eps=1 px (contours with area < 100 px are not simplified, 1-px components become a pixel box).
- Full conversion (`runs/checks/yolo/stats.csv`): train 2345 images / 31994 polygons, mean change 0.062% (max 0.41%), min thing IoU 0.81; val mean change 0.069%. Ultralytics loads all labels, 0 corrupt.
- Conclusion: the conversion itself costs < 0.1% of pixels, so any YOLO vs SegFormer gap is from the model/representation, not the label format. Without keyhole, objects with holes (e.g. boats with see-through parts) lose up to half their mask.
- YOLO training choices: mosaic off (2x2 mosaic puts sky below water), no vertical flip, rotation +-5 deg.
