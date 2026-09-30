import cv2
import numpy as np
import pytest

from aerometric.config import SegmentationConfig
from aerometric.segmentation import (
    Label,
    Segmenter,
    SegmentationResult,
    load_labels,
    make_segmenter,
    save_labels,
)
from aerometric.types import DataError

MOCK = SegmentationConfig(backend="mock")


def scene():
    """Grey textured ground with a green field on the left and a smooth blue pond on the right."""
    rng = np.random.default_rng(0)
    img = (rng.random((300, 600, 3)) * 60 + 100).astype(np.uint8)
    img[50:250, 20:200] = (40, 160, 40)  # BGR green
    img[50:250, 20:200] += rng.integers(0, 20, (200, 180, 3), dtype=np.uint8)
    img[80:220, 380:560] = (180, 90, 30)  # BGR blue, flat
    return img


def test_mock_is_a_segmenter_and_says_so():
    seg = make_segmenter(MOCK)
    assert isinstance(seg, Segmenter)
    assert seg.is_mock and seg.name == "mock"


def test_mock_labels_vegetation_and_water():
    res = make_segmenter(SegmentationConfig(backend="mock", work_width=300)).segment(scene())
    assert res.labels.shape == (300, 600)
    assert res.is_mock and np.all(res.confidence == pytest.approx(0.3))
    assert np.mean(res.labels[70:230, 40:180] == Label.VEGETATION) > 0.9
    assert np.mean(res.labels[100:200, 400:540] == Label.WATER) > 0.9
    assert np.mean(res.labels[0:40, 250:350] == Label.OTHER) > 0.9
    assert not res.mask_of({Label.PERSON, Label.VEHICLE}).any()


def test_exclusion_mask_covers_water_and_dilates():
    res = make_segmenter(MOCK).segment(scene())
    plain = res.exclusion_mask()
    grown = res.exclusion_mask(dilate_px=5)
    assert plain[150, 470] and not plain[150, 100]  # water excluded, vegetation kept
    assert grown.sum() > plain.sum()
    edge = int(np.argmax(plain[150]))  # first excluded column on this row
    assert grown[150, edge - 3] and not plain[150, edge - 3]
    assert not grown[150, edge - 8]


def test_dynamic_classes_are_excluded():
    labels = np.zeros((40, 40), np.uint8)
    labels[10:20, 10:20] = Label.VEHICLE
    labels[25:30, 25:30] = Label.BUILDING
    res = SegmentationResult(labels, None, "test", False)
    mask = res.exclusion_mask()
    assert mask[15, 15] and not mask[27, 27]
    assert res.class_fractions()["vehicle"] == pytest.approx(100 / 1600)


def test_contract_validation():
    with pytest.raises(DataError, match="uint8"):
        SegmentationResult(np.zeros((4, 4), np.int32), None, "x", False)
    with pytest.raises(DataError, match="confidence"):
        SegmentationResult(np.zeros((4, 4), np.uint8), np.zeros((2, 2)), "x", False)
    with pytest.raises(DataError, match="colour"):
        make_segmenter(MOCK).segment(np.zeros((10, 10), np.uint8))


def test_unknown_backend():
    with pytest.raises(DataError, match="not available"):
        make_segmenter(SegmentationConfig(backend="segformer"))


def test_label_cache_roundtrip(tmp_path):
    res = make_segmenter(MOCK).segment(scene())
    save_labels(res, tmp_path / "labels" / "000120")
    back = load_labels(tmp_path / "labels" / "000120", res.backend, res.is_mock)
    assert np.array_equal(back.labels, res.labels)
    assert back.is_mock

