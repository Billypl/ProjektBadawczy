import numpy as np

from common.interfaces import IGNORE, OBSTACLE, SKY, WATER
from usvseg.train.transforms import TrainTransform, keep_ratio_scale


def test_keep_ratio_scale_matches_mmseg_rule():
    assert keep_ratio_scale((1080, 1920), (2048, 1024)) == min(2048 / 1920, 1024 / 1080)


def test_train_transform_shapes_and_label_values():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    mask = np.full((720, 1280), WATER, np.uint8)
    mask[:300] = SKY
    mask[280:320, 500:700] = OBSTACLE
    mask[700:] = IGNORE
    t = TrainTransform(crop_hw=(256, 512), rotate_prob=1.0)
    for seed in range(20):
        x, y = t(img, mask, np.random.default_rng(seed))
        assert tuple(x.shape) == (3, 256, 512) and tuple(y.shape) == (256, 512)
        assert set(np.unique(y.numpy()).tolist()) <= {OBSTACLE, WATER, SKY, IGNORE}


def test_sky_stays_above_water():
    """No vertical flip: in every crop sky rows lie above water rows."""
    img = np.zeros((400, 800, 3), np.uint8)
    mask = np.full((400, 800), WATER, np.uint8)
    mask[:200] = SKY
    t = TrainTransform(base_scale=(800, 400), crop_hw=(300, 600), rotate_deg=0)
    for seed in range(20):
        _, y = t(img, mask, np.random.default_rng(seed))
        y = y.numpy()
        sky_rows, water_rows = np.where(y == SKY)[0], np.where(y == WATER)[0]
        if len(sky_rows) and len(water_rows):
            assert sky_rows.mean() < water_rows.mean()
