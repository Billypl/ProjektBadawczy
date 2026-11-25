import numpy as np

from common.interfaces import OBSTACLE, SKY, WATER
from usvseg.postprocess import BlobFilter, remove_small_blobs


def scene() -> np.ndarray:
    labels = np.full((200, 300), WATER, np.uint8)
    labels[:50] = SKY
    labels[50:80] = OBSTACLE           # shore, large
    labels[85:88, 100:103] = OBSTACLE  # small blob 5 px below the shore
    labels[150:153, 200:203] = OBSTACLE  # small blob far from the shore
    labels[120:150, 20:60] = OBSTACLE  # large boat, must stay
    return labels


def test_shore_only_removes_near_blob():
    out = remove_small_blobs(scene(), BlobFilter(max_area=50, shore_dist=10))
    assert (out[85:88, 100:103] == WATER).all()
    assert (out[150:153, 200:203] == OBSTACLE).all()
    assert (out[120:150, 20:60] == OBSTACLE).all()
    assert (out[50:80] == OBSTACLE).all()


def test_anywhere_removes_both_blobs():
    out = remove_small_blobs(scene(), BlobFilter(max_area=50, shore_dist=0))
    assert (out[85:88, 100:103] == WATER).all()
    assert (out[150:153, 200:203] == WATER).all()
    assert (out[120:150, 20:60] == OBSTACLE).all()


def test_blob_next_to_sky_is_kept():
    labels = np.full((100, 100), WATER, np.uint8)
    labels[:40] = SKY
    labels[37:40, 50:53] = OBSTACLE  # small object on the horizon, bordered by sky
    out = remove_small_blobs(labels, BlobFilter(max_area=50, shore_dist=0))
    assert (out[37:40, 50:53] == OBSTACLE).all()
