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

## 2026-11-08 - M1: SegFormer-B2 trained on LaRS

- What: HF `SegformerForSemanticSegmentation`, encoder `nvidia/mit-b2` (ImageNet-1k; NVIDIA Source Code License, research only). 30 epochs on LaRS train minus dev (2345 images), batch 8, crop 512x1024, keep-ratio scale (2048, 1024) x [0.5, 2.0], cat_max_ratio 0.75, hflip, colour jitter, rotation +-5 deg; AdamW lr 6e-5, head lr x10, poly with 879-iter warmup, bf16. Inference: whole image, keep-ratio (2048, 1024), last checkpoint. Run: `runs/m1_segformer_b2/` (config, metrics.csv, train.log).
- Why: main semantic model for H1; hyperparameters taken from the MaCVi LaRS starter kit, schedule cut from 160k iterations to 30 epochs (8790 iterations).
- Training: about 1.8 h on the RTX 5070 Ti (epochs 1-25 ~240 s, partly slowed by parallel CPU work; epochs 26-30 ~115 s). Dev mIoU 95.6 (ep 5) -> 97.2 (ep 20) -> 97.5 (ep 30), still slowly rising.
- Result (lars_evaluator, val): **F1 69.8, mu 76.4, mIoU 96.7, Q 67.5**. TP 943, FP 653, FN 164 -> precision 59.1, recall 85.2. Close to the published SegFormer (MiT-B2) LaRS test numbers (70.0 / 78.6 / 96.8 / 67.8).
- Breakdown: river_like F1 37.9 (Pr 24.5) vs sea_like F1 80.1; reflections none 75.0 / moderate 69.1 / heavy 2.3 (n=10).
- Conclusion: vs M0 eWaSR +20.5 F1, +17.1 mu. FP halved (1450 -> 653), but rivers and heavy reflections remain the weak spot: river precision is still only 24.5%. Longer training may help a little (dev mIoU still rising), but the river FP problem looks structural -> H2 (stereo filtering of reflections) and H3.

## 2026-11-08 - M2: YOLO26s-seg trained on LaRS

- What: `yolo26s-seg.pt` (COCO) fine-tuned 50 epochs on the converted LaRS (train minus dev, Ultralytics val = dev), imgsz 1024, batch 8, mosaic off, no vertical flip, rotation +-5 deg. best.pt chosen by Ultralytics on dev. Inference conf 0.25, retina masks, then instance -> semantic map (obstacles over sky over water). Run: `runs/m2_yolo26s_seg/` (38 min).
- Why: instance-segmentation baseline for H1.
- Dev (Ultralytics): mask mAP50 0.53, mAP50-95 0.30.
- Unassigned pixels on val: mean 1.67%, max 22.1% of the image.
- Result (lars_evaluator, val):

| variant | F1 | mu | mIoU | Q | TP | FP | FN | Pr | Re |
|---|---|---|---|---|---|---|---|---|---|
| unassigned -> obstacle | 71.6 | 74.0 | 95.4 | 68.3 | 999 | 685 | 108 | 59.3 | 90.2 |
| unassigned -> water | 73.2 | 74.4 | 95.4 | 69.8 | 861 | 386 | 246 | 69.0 | 77.8 |

- The two variants trade FP for FN: unassigned pixels often sit around real obstacles, so mapping them to water removes about 300 FP and adds about 140 FN. For navigation the cautious variant is the relevant one.

## 2026-11-08 - Summary M0-M2 (LaRS val, 198 images)

| ID | model | train data | F1 | mu | mIoU | Q |
|---|---|---|---|---|---|---|
| M0 | eWaSR ResNet-18 (no IMU) | MaSTr1325 | 49.3 | 59.3 | 85.1 | 42.0 |
| M1 | SegFormer-B2, 30 ep | LaRS train (-dev) | 69.8 | **76.4** | **96.7** | 67.5 |
| M2 | YOLO26s-seg, 50 ep, unassigned -> obstacle | LaRS train (-dev) | 71.6 | 74.0 | 95.4 | 68.3 |
| M2 | YOLO26s-seg, 50 ep, unassigned -> water | LaRS train (-dev) | **73.2** | 74.4 | 95.4 | **69.8** |

Source: `results/results.csv` (evaluator output). F1 by scene type: M0 river 24.3 / sea 60.6, M1 river 37.9 / sea 80.1, M2 (obstacle) river 39.8 / sea 82.0.

Paired bootstrap over frames (5000 resamples of the evaluator's per-frame TP/FP/FN and WE_acc), YOLO minus SegFormer:
- F1: +1.8 (95% CI -4.1 to +6.8) for the cautious variant, +3.4 (CI -3.3 to +9.2) for the water variant -> **not significant**.
- per-frame mu: -2.2 (CI -3.2 to -1.1) and -1.7 (CI -2.7 to -0.7) -> **SegFormer's water edge is significantly better**.

Conclusions so far (H1):
- Water-edge accuracy (mu) supports H1: the semantic model draws the water/obstacle boundary better (also mIoU 96.7 vs 95.4).
- Obstacle F1 does **not** support H1 yet: YOLO is on par or slightly better, mostly through higher recall (90.2 vs 85.2). The F1 difference is within noise on 198 frames.
- Not a fair final comparison yet: SegFormer had 30 epochs vs 160k iterations (~500 epochs) in the starter kit and its dev mIoU was still rising; YOLO used best-on-dev, SegFormer the last checkpoint. FPS (part of H1) is not measured yet.
- All models fail on rivers compared to sea (precision 25-30% on rivers); heavy reflections remain unsolved (F1 < 10 for every model, n=10).

Observed errors (`results/figures/comparison/`, gitignored, contains LaRS images):
- Sun glitter (`inhouse_seq_155_00012`): eWaSR 32 FP around the glitter path, SegFormer 4, YOLO 2.
- Brown jungle river (`yt055_02_00114`): eWaSR breaks down completely, both trained models are close to GT.
- Reflections of trees on calm rivers (`inhouse_seq_244_*`): main remaining FP source, matches H2.

Next: FPS for M0-M2 (batch 1, 50 warm-up + 200 timed iterations); longer SegFormer schedule; check whether a lower YOLO conf threshold changes the F1/mu balance (tune on dev, not val).

## 2026-11-08 - FPS, checkpoint selection, YOLO conf threshold

### FPS (`usvseg/eval/fps.py`, `results/fps.csv`)

- What: `Segmenter.segment()` end to end (RGB image in memory -> labels + confidence at the original resolution, incl. resize, normalisation, upsampling, host<->device), batch 1, 50 warm-up + 200 timed iterations with `torch.cuda.synchronize`, cycling through 20 val images (1920x1080 and 2208x1242). RTX 5070 Ti, bf16 for SegFormer, fp32 for eWaSR, Ultralytics default (fp32) for YOLO.
- Why: H1 compares the models "at comparable FPS".

| model | input | mean ms | p95 ms | FPS |
|---|---|---|---|---|
| M0 eWaSR | 512x384 | 27.1 | 30.7 | 36.9 |
| M1 SegFormer-B2 | keep-ratio 2048x1024 | 70.4 | 73.3 | 14.2 |
| M1 SegFormer-B2 | keep-ratio 1536x768 | 41.4 | 43.1 | 24.1 |
| M1 SegFormer-B2 | keep-ratio 1024x512 | 21.4 | 23.8 | 46.8 |
| M2 YOLO26s-seg | imgsz 1024, conf 0.25 | 28.0 | 36.1 | 35.7 |
| M2 YOLO26s-seg | imgsz 1024, conf 0.10 | 36.0 | 52.3 | 27.8 |

- Conclusion: at its default scale SegFormer is 2-2.5x slower than YOLO, so the M1 vs M2 comparison above was not at comparable FPS. YOLO time grows with the number of instances (lower conf -> more masks to upsample, larger p95). Pre/post-processing is not optimised in any wrapper (full-resolution softmax, CPU normalisation), so these are pipeline numbers, not model-only numbers. All of it is on a desktop GPU; on the boat the numbers will differ.

### Checkpoint selection (`results/dev_selection.csv`)

- What: one rule for both models: pick the checkpoint with the highest evaluator Q on dev.
- SegFormer saves only the last checkpoint, but epoch 30 is also its best dev mIoU (monotonic rise, see `runs/m1_segformer_b2/metrics.csv`), so last = best. Dev: F1 67.2, mu 78.2, Q 65.6.
- YOLO `best.pt` (epoch 48, chosen by Ultralytics mAP fitness) vs `last.pt` (epoch 50), conf 0.25, dev Q: 71.2 vs 70.6 (unassigned -> obstacle), 77.9 vs 77.3 (-> water).
- Conclusion: keep `best.pt` for YOLO and the last checkpoint for SegFormer; the earlier val numbers stand. Note: dev was also Ultralytics' validation set, so YOLO's best.pt is mildly optimistic on dev (not on val).

### YOLO confidence threshold (dev, `best.pt`)

| conf | Q (-> obstacle) | Q (-> water) |
|---|---|---|
| 0.05 | 71.5 | 81.4 |
| 0.10 | **71.6** | **81.4** |
| 0.15 | 71.6 | 81.0 |
| 0.25 | 71.2 | 77.9 |
| 0.35 | 70.9 | 74.4 |
| 0.50 | 69.5 | 65.8 |

- Chosen: conf 0.10 for both variants (0.05 ties, the higher threshold gives fewer instances). A lower threshold mostly helps the water variant: more pixels get covered by some mask, so fewer obstacle pixels get "washed" into water.
- Val (run once with the chosen setting): see the updated table below.

### Updated H1 comparison (LaRS val, 198 images)

| model | FPS | F1 | mu | mIoU | Q | Pr | Re |
|---|---|---|---|---|---|---|---|
| SegFormer-B2, scale 2048x1024 | 14.2 | 69.8 | 76.4 | 96.7 | 67.5 | 59.1 | 85.2 |
| SegFormer-B2, scale 1536x768 | 24.1 | 70.1 | 74.7 | 96.9 | 67.9 | 61.8 | 80.9 |
| SegFormer-B2, scale 1024x512 | 46.8 | 66.3 | 71.5 | 96.8 | 64.1 | 61.1 | 72.4 |
| YOLO26s-seg conf 0.10, unassigned -> obstacle | 27.8 | 72.6 | 74.6 | 95.8 | 69.5 | 59.9 | 92.1 |
| YOLO26s-seg conf 0.10, unassigned -> water | 27.8 | 78.1 | 75.0 | 95.7 | 74.7 | 70.3 | 87.8 |

Source: `results/results.csv` and `results/fps.csv`. SegFormer's inference scale was not tuned; 1536x768 is simply the scale closest to YOLO's FPS.

Paired bootstrap (`scripts/bootstrap_compare.py`, 5000 resamples, YOLO conf 0.10 minus SegFormer):

| SegFormer scale | YOLO variant | F1 diff (95% CI) | mu diff (95% CI) |
|---|---|---|---|
| 2048x1024 (14 FPS) | -> obstacle | +2.8 (-3.2, +7.9) | -1.3 (-2.2, -0.4) |
| 2048x1024 (14 FPS) | -> water | **+8.3 (+1.5, +14.0)** | -0.8 (-1.7, +0.1) |
| 1536x768 (24 FPS, ~ YOLO) | -> obstacle | +2.4 (-3.5, +7.3) | +0.3 (-0.6, +1.2) |
| 1536x768 (24 FPS, ~ YOLO) | -> water | **+8.0 (+1.3, +13.4)** | +0.8 (-0.1, +1.8) |

Conclusions (H1, current state):
- At comparable FPS, H1 is **not supported**: SegFormer is not better on F1 nor on mu. YOLO with unassigned -> water is significantly better on F1 (+8); the cautious YOLO variant is on par on both metrics.
- SegFormer's mu advantage only exists at its full scale, where it runs at half of YOLO's FPS; tuning YOLO's threshold also shrank that gap from -2.2 to -1.3.
- Caveats: SegFormer was trained 30 epochs (starter kit ~500) with dev mIoU still rising, and only YOLO had a threshold to tune. The unassigned -> water variant is optimistic: it turns "the model did not say anything here" into "safe water", which is risky for navigation.
- Next: longer SegFormer training (the main open question for H1), then possibly a SegFormer trained/inferred at a smaller scale to match FPS properly.

## 2026-11-09 - River error analysis (for H3)

- What: `scripts/river_analysis.py` -> `results/river_analysis.md`. Breakdown river_like vs sea_like on LaRS val for M0 eWaSR, M1 SegFormer-B2 (30 ep, full scale) and M2 YOLO26s-seg (conf 0.10, unassigned -> obstacle), from the evaluator's frames.csv and segments.json plus LaRS scene attributes. 10 overlays of the worst river frames in `results/figures/rivers/` (gitignored).
- Why: all models had river precision of only 14-26%; H3 needs to know whether this is "the model is worse on rivers" or something else.
- Data mix: LaRS train (minus dev) is 77% sea / 23% river (1815 / 529); val is 60 / 40 (119 / 79). MaSTr1325 inside LaRS train is all sea.
- Results:
  - **Rivers have few real obstacles.** River frames contain 149 GT obstacles (1.9 per frame), sea frames 958 (8.1 per frame). Recall on rivers is fine (84-95%, YOLO 94.6%); the low F1 comes from precision.
  - **FP per frame is about 2x higher on rivers** for both trained models (SegFormer 4.9 vs 2.2, YOLO 5.2 vs 2.3; eWaSR 9.8 vs 5.7). Combined with 4x fewer real obstacles this collapses precision.
  - **Still water and heavy reflections:** 21 still-water river frames contain only 3 GT obstacles in total, so F1 there (1-3) is meaningless; FP/frame (4.4-9.6) is the metric to watch. The "heavy reflections" subset (n=10) has the same problem. Previous notes reading "F1 < 10 on heavy reflections" overstate it: it is a handful of FP on frames with almost nothing to detect.
  - FP segments are mostly small (<100 px: 61-67% for the trained models on rivers), but rivers have more large ones for eWaSR (>10k px: 10% vs 3%).
  - Location of false obstacle pixels (own diagnostic: predicted obstacle on GT water, by distance to the nearest non-water GT pixel): SegFormer on rivers has 37% within 10 px of the shore (edge noise) but **31.5% more than 150 px away** (vs 7% on sea), i.e. blobs in open water: reflections and glitter. YOLO on rivers is more spread (40% near shore, 15% far).
  - FN: on rivers almost nothing is missed (YOLO 8 FN of 149). On sea the hard classes are animals (57-96% missed), buoys (18-45%) and "other".
- Observed error types (worst river frames):
  - Mirror reflections of trees on still ponds (`yt098_02_*`): SegFormer produces many small FP blobs on the reflections, eWaSR labels most of the reflection as obstacle, YOLO is almost clean.
  - Sun glitter against the light (`yt099_19_*`, fjord labelled river_like): FP on glitter and on dark water in the mountain's shadow.
  - Dark rippled water in an overexposed scene under a bridge (`mit_seq_08_00219`): YOLO 85 FP of small patches on high-contrast ripples, SegFormer 2.
  - Shore line of narrow rivers (`davimar_seq_70_00039`): small FP along the water/bank boundary.
  - Own boat parts at the bottom of the image are sometimes annotated as water in GT (`yt099_19_00029`), so a correct "obstacle" prediction there counts as FP. Not quantified.
- Conclusions:
  - H3 in its plain form ("model trained mostly on sea loses on rivers") is supported by FP/frame (2x), but the F1/precision gap overstates it, because river scenes have far fewer obstacles. For H3 report FP/frame (or FP per km/minute on own recordings) alongside F1.
  - The river FP split into two groups with different fixes: edge noise near the shore (post-processing, minimum blob size, temporal filtering) and blobs far out in open water from reflections and glitter (exactly H2: a stereo point on a reflection lies below the water plane).
  - Neither model type is uniformly better on rivers: SegFormer is cleaner on bright/overexposed scenes, YOLO on mirror reflections.

## 2026-11-12 - M1 SegFormer-B2, 80 epochs

- What: same setup as M1 (`configs/m1_segformer_b2_e80.yaml`), 80 epochs instead of 30 (23440 iterations; warmup 1500 iterations, since total/10 no longer caps it). Run `runs/m1_segformer_b2_e80/`, interrupted and resumed twice (after epochs 4 and 21) with `--resume`. About 3.5 h of training in total.
- Why: dev mIoU of the 30-epoch run was still rising; H1 needed a fairer SegFormer.
- Training: dev mIoU 94.5 (ep 5) -> 97.2 (ep 25) -> 97.5 (ep 35) -> 97.6 (ep 50-65) -> 97.87 (ep 80). Last epoch = best dev mIoU, so `checkpoint/` and `checkpoint_best/` are the same model.
- Dev (evaluator): F1 73.2, mu 79.5, Q 71.6 (30 epochs: 67.2 / 78.2 / 65.6).
- Val (evaluator, `results/results.csv`):

| model | scale | FPS | F1 | mu | mIoU | Q | Pr | Re |
|---|---|---|---|---|---|---|---|---|
| SegFormer-B2 30 ep | 2048x1024 | 14.2 | 69.8 | 76.4 | 96.7 | 67.5 | 59.1 | 85.2 |
| SegFormer-B2 80 ep | 2048x1024 | 14.2 | 71.6 | **77.4** | 97.1 | 69.5 | 61.2 | 86.1 |
| SegFormer-B2 80 ep | 1536x768 | 24.1 | 72.4 | 75.7 | **97.2** | 70.4 | 64.0 | 83.4 |
| YOLO26s-seg conf 0.10, -> obstacle | 1024 | 27.8 | 72.6 | 74.6 | 95.8 | 69.5 | 59.9 | 92.1 |
| YOLO26s-seg conf 0.10, -> water | 1024 | 27.8 | **78.1** | 75.0 | 95.7 | **74.7** | 70.3 | 87.8 |

FPS is taken from the 30-epoch measurements (`results/fps.csv`): same architecture and inference code, so the speed does not change.

- 80 vs 30 epochs (full scale, paired bootstrap): F1 +1.8 (CI -1.0, +4.9, within noise), mu +0.9 (CI +0.5, +1.3, significant).
- Rivers (full scale): F1 41.2 (30 ep: 37.9), FP 356 (30 ep: 386); sea F1 81.1 (80.1).

Paired bootstrap, YOLO conf 0.10 minus SegFormer 80 ep:

| SegFormer scale | YOLO variant | F1 diff (95% CI) | mu diff (95% CI) |
|---|---|---|---|
| 2048x1024 (14 FPS) | -> obstacle | +1.0 (-5.0, +6.2) | **-2.2 (-3.1, -1.3)** |
| 2048x1024 (14 FPS) | -> water | +6.5 (-0.3, +12.3) | **-1.7 (-2.6, -0.8)** |
| 1536x768 (24 FPS, ~ YOLO) | -> obstacle | +0.1 (-5.5, +4.5) | -0.6 (-1.5, +0.3) |
| 1536x768 (24 FPS, ~ YOLO) | -> water | +5.7 (-0.7, +10.5) | -0.1 (-1.0, +0.9) |

Conclusions (H1, current state):
- Longer training helped SegFormer a little (mu significantly, F1 within noise) and removed YOLO's significant F1 lead; the remaining F1 gap of the optimistic YOLO variant (+5.7 to +6.5) is borderline (CI just touches 0).
- At comparable FPS (~24-28) **neither model is significantly better on F1 or mu**. H1 is not supported, but not refuted either: the difference is below what 198 val frames can resolve.
- SegFormer's water-edge advantage (mu +1.7 to +2.2, significant) exists only at full scale, i.e. at about half of YOLO's FPS.
- Practical takeaway for the boat: model choice matters less than expected; YOLO gives instances for free (useful for tracking and for the stereo part), SegFormer gives a slightly better water edge. Decision should depend on the target hardware FPS and on how the stereo part uses the output.
- To resolve the F1 question, more evaluation frames are needed (LaRS test via the MaCVi server, or own annotated recordings), not more training.
