# USV segmentation (ZPB, PG) - water / obstacle / sky

Billy's part of the ZPB project: dataset selection, semantic segmentation models and evaluation on LaRS.
See `CLAUDE.md` for decisions, conventions and the plan.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -e ".[dev]" yacs==0.1.8

# third-party repos (gitignored)
mkdir -p third_party && cd third_party
git clone https://github.com/tersekmatija/eWaSR.git
git clone https://github.com/lojzezust/lars_evaluator.git
# eWaSR has its own venv (old timm pin), weights from the 0.1.0 release
cd eWaSR && python3 -m venv .venv
.venv/bin/pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install timm==0.6.5 pytorch-lightning Pillow numpy tqdm albumentations pyyaml
mkdir -p pretrained && curl -L -o pretrained/ewasr_resnet18.pth \
  https://github.com/tersekmatija/eWaSR/releases/download/0.1.0/ewasr_resnet18.pth
```

## Data

Download LaRS v1.0.0 (`lars_v1.0.0_images.zip`, `lars_v1.0.0_annotations.zip`) from
https://lojzezust.github.io/lars-dataset and unzip both into `data/raw/lars/`.

## Usage

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/lars_stats.py                 # stats + overlays in runs/checks/lars/
third_party/eWaSR/.venv/bin/python scripts/predict_ewasr.py --exp m0_ewasr_val --split val
.venv/bin/python -m usvseg.eval.run_lars_eval --exp m0_ewasr_val --model ewasr_resnet18 \
    --train-data MaSTr1325 --split val --input-res 512x384
```

## Class ids

`OBSTACLE, WATER, SKY, IGNORE = 0, 1, 2, 255` (LaRS / MaSTr convention), see `common/interfaces.py`.
