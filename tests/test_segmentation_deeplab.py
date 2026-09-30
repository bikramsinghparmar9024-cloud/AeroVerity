import cv2
import numpy as np
import pytest
from skimage import data

from dronerecon.config import SegmentationConfig
from dronerecon.segmentation import Label, make_segmenter
from dronerecon.segmentation.deeplab import weights_cached

pytestmark = pytest.mark.skipif(not weights_cached(), reason="DeepLabV3 weights not in the local torch cache")


@pytest.fixture(scope="module")
def seg():
    return make_segmenter(SegmentationConfig(backend="deeplab", work_width=520))


def bgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def test_auto_prefers_the_learned_model():
    s = make_segmenter(SegmentationConfig(backend="auto"))
    assert not s.is_mock and s.name.startswith("deeplabv3")


def test_finds_a_person(seg):
    img = bgr(data.astronaut())  # 512 x 512 portrait photo
    res = seg.segment(img)
    assert res.labels.shape == img.shape[:2] and not res.is_mock
    person = res.mask_of({Label.PERSON})
    assert person.mean() > 0.15  # the astronaut fills a good part of the frame
    assert person[200:400, 200:350].mean() > 0.7  # torso region
    assert res.exclusion_mask()[300, 280]  # people are kept out of matching
    assert np.nanmean(res.confidence[person]) > 0.6


def test_finds_an_animal(seg):
    res = seg.segment(bgr(data.chelsea()))  # a cat
    assert res.mask_of({Label.ANIMAL}).mean() > 0.2


def test_plain_ground_has_no_moving_objects(seg):
    rng = np.random.default_rng(0)
    ground = cv2.GaussianBlur((rng.random((480, 640, 3)) * 80 + 90).astype(np.uint8), (0, 0), 3)
    res = seg.segment(ground)
    assert res.mask_of({Label.PERSON, Label.VEHICLE, Label.ANIMAL}).mean() < 0.01
