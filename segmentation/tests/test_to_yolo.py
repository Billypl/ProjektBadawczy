import numpy as np

from common.interfaces import OBSTACLE, SKY, WATER
from usvseg.data.to_yolo import (LARS_TO_YOLO, convert_image, from_yolo_lines, mask_to_polygons, rasterize,
                                 semantic_from_class_masks, to_yolo_lines)


def _scene():
    """60x80: sky on top, water below, a boat (with a hole) and a 1-px buoy in the water."""
    h, w = 60, 80
    pan = np.full((h, w), 3, np.uint32)          # water segment id 3
    pan[:20] = 5                                  # sky segment id 5
    pan[30:45, 20:50] = 65547                     # boat, category 11
    pan[35:40, 30:40] = 3                         # water visible through the boat (hole)
    pan[50, 70] = 65550                           # 1-pixel buoy, category 14
    segs = [{"id": 3, "category_id": 3}, {"id": 5, "category_id": 5},
            {"id": 65547, "category_id": 11}, {"id": 65550, "category_id": 14}]
    sem = np.full((h, w), WATER, np.uint8)
    sem[:20] = SKY
    sem[pan % 256 >= 11] = OBSTACLE
    return pan, segs, sem


def test_round_trip_exact_without_simplification():
    pan, segs, sem = _scene()
    polys, stats = convert_image(pan, segs, sem, keyhole=True, epsilon=0)
    back = from_yolo_lines(to_yolo_lines(polys, pan.shape), pan.shape)
    rebuilt = semantic_from_class_masks(rasterize(back, pan.shape), pan.shape)
    changed = (rebuilt != sem).mean() * 100
    assert changed < 1.0, changed
    assert stats.n_tiny == 1                      # buoy kept as a pixel box
    assert rebuilt[50, 70] == OBSTACLE
    assert min(stats.thing_iou) > 0.9


def test_keyhole_keeps_hole_empty():
    pan, segs, _ = _scene()
    boat = pan == 65547
    with_cut, _, _ = mask_to_polygons(boat, keyhole=True, epsilon=0)
    without, n_holes, _ = mask_to_polygons(boat, keyhole=False, epsilon=0)
    assert n_holes == 1
    from usvseg.data.to_yolo import Polygon
    r_cut = rasterize([Polygon(3, p) for p in with_cut], boat.shape)[3]
    r_fill = rasterize([Polygon(3, p) for p in without], boat.shape)[3]
    assert r_fill[37, 35] and not r_cut[37, 35]   # hole centre filled only without keyhole
    iou = (r_cut & boat).sum() / (r_cut | boat).sum()
    assert iou > 0.95


def test_class_mapping():
    assert LARS_TO_YOLO[1] == 0 and LARS_TO_YOLO[3] == 1 and LARS_TO_YOLO[5] == 2
    assert len(LARS_TO_YOLO) == 11
