import json

import numpy as np
import pytest
from PIL import Image

from common.interfaces import IGNORE, OBSTACLE, SKY, WATER, StereoFrame
from usvseg.data.lars import Lars, dev_names
from usvseg.data.taxonomy import CLASS_COLORS, color_to_labels, labels_to_color, mastr_to_semantic
from usvseg.models.gt import GroundTruthSegmenter


def test_evaluator_colors_match_docs():
    assert CLASS_COLORS[SKY] == (90, 75, 164)
    assert CLASS_COLORS[WATER] == (41, 167, 224)
    assert CLASS_COLORS[OBSTACLE] == (247, 195, 37)


def test_color_round_trip():
    labels = np.array([[OBSTACLE, WATER], [SKY, IGNORE]], dtype=np.uint8)
    assert np.array_equal(color_to_labels(labels_to_color(labels)), labels)


def test_mastr_ignore_is_remapped():
    mask = np.array([[0, 1, 2, 4]], dtype=np.uint8)
    assert mastr_to_semantic(mask).tolist() == [[0, 1, 2, 255]]


def test_mastr_rejects_unknown_values():
    with pytest.raises(ValueError):
        mastr_to_semantic(np.array([[3]], dtype=np.uint8))


def test_dev_split_is_deterministic_and_disjoint():
    names = [f"img_{i:04d}" for i in range(100)]
    a, b = dev_names(names), dev_names(list(reversed(names)))
    assert a == b and len(a) == 10


def _fake_lars(root, n=20):
    split = root / "lars_v1.0.0_images" / "train"
    ann = root / "lars_v1.0.0_annotations" / "train"
    (split / "images").mkdir(parents=True)
    (ann / "semantic_masks").mkdir(parents=True)
    (ann / "panoptic_masks").mkdir(parents=True)
    annotations = []
    for i in range(n):
        name = f"f{i:03d}"
        Image.fromarray(np.zeros((4, 6, 3), np.uint8)).save(split / "images" / f"{name}.jpg")
        sem = np.array([[2] * 6, [2] * 6, [0, 1, 1, 1, 1, 0], [1] * 6], np.uint8)
        Image.fromarray(sem).save(ann / "semantic_masks" / f"{name}.png")
        seg_id = 300  # -> R=44, G=1
        pan = np.zeros((4, 6, 3), np.uint8)
        pan[2, 0] = (seg_id % 256, seg_id // 256, 0)
        Image.fromarray(pan).save(ann / "panoptic_masks" / f"{name}.png")
        annotations.append({"file_name": f"{name}.png",
                            "segments_info": [{"id": seg_id, "category_id": 11}]})
    (ann / "panoptic_annotations.json").write_text(json.dumps(
        {"annotations": annotations, "categories": [{"id": 11, "name": "Boat/ship", "isthing": 1}]}))


def test_lars_train_dev_partition(tmp_path):
    _fake_lars(tmp_path)
    train, dev = Lars("train", tmp_path), Lars("dev", tmp_path)
    assert len(train) + len(dev) == 20 and len(dev) == 2
    assert not {s.name for s in train} & {s.name for s in dev}


def test_gt_segmenter(tmp_path):
    _fake_lars(tmp_path)
    ds = Lars("train", tmp_path)
    frame = StereoFrame(left=ds[0].image(), right=None, K=None, baseline_m=None,
                        timestamp=0.0, frame_id=ds[0].name)
    seg = GroundTruthSegmenter(ds).segment(frame)
    assert seg.labels.shape == (4, 6) and seg.labels[0, 0] == SKY
    assert len(seg.instances) == 1 and seg.instances[0].mask.sum() == 1
    assert seg.instances[0].mask[2, 0]


def test_save_prediction_resizes_nearest_to_original(tmp_path):
    from usvseg.eval.export_lars import save_prediction

    labels = np.array([[SKY, SKY], [OBSTACLE, WATER]], np.uint8)
    path = save_prediction(labels, tmp_path, "x", (4, 6))
    back = color_to_labels(np.asarray(Image.open(path).convert("RGB")))
    assert back.shape == (4, 6)
    assert set(np.unique(back).tolist()) == {OBSTACLE, WATER, SKY}
