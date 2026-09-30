import cv2
import numpy as np
import pytest

from aerometric.config import MatchingConfig
from aerometric.matching.features import (
    LightGlueBackend,
    lightglue_available,
    make_matching,
    verify_pair,
)
from aerometric.types import DataError

OPENCV = MatchingConfig(backend="opencv", max_dim=800)


def ground(seed=0, w=1400, h=900):
    rng = np.random.default_rng(seed)
    small = (rng.random((h // 8, w // 8, 3)) * 255).astype(np.uint8)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(60):  # add corners/edges like buildings and roads
        x, y = rng.integers(0, w - 60), rng.integers(0, h - 60)
        cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(15, 60)), int(y + rng.integers(15, 60))),
                      tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return img


def view_pair(dx=90.0, dy=25.0, angle=4.0):
    """Two overlapping views related by a known similarity transform (planar ground)."""
    g = ground()
    A = cv2.getRotationMatrix2D((700, 450), 0, 1.0)
    B = cv2.getRotationMatrix2D((700, 450), angle, 1.0)
    B[:, 2] += (dx, dy)
    a = cv2.warpAffine(g, A, (1200, 800))
    b = cv2.warpAffine(g, B, (1200, 800))
    return a, b, B


def test_auto_uses_fallback_when_lightglue_missing():
    ext, mat = make_matching(MatchingConfig(backend="auto"))
    if lightglue_available():
        pytest.skip("lightglue installed; fallback path not exercised")
    assert ext.is_fallback and mat.is_fallback
    assert ext.name == "sift"


def test_lightglue_backend_explains_missing_package():
    if lightglue_available():
        pytest.skip("lightglue installed")
    with pytest.raises(DataError, match="not installed"):
        LightGlueBackend(MatchingConfig(backend="lightglue"))


def test_bad_backend_name():
    with pytest.raises(DataError, match="backend"):
        make_matching(MatchingConfig(backend="superglue"))


def test_fallback_matches_are_geometrically_correct():
    a, b, B = view_pair()
    ext, mat = make_matching(OPENCV)
    f0, f1 = ext.extract(a), ext.extract(b)
    assert len(f0) > 500 and f0.image_size == (1200, 800)
    pair = verify_pair(f0, f1, mat.match(f0, f1), OPENCV)
    assert len(pair.matches) > 200 and pair.inlier_ratio > 0.6
    p0 = f0.keypoints[pair.matches.idx0]
    p1 = f1.keypoints[pair.matches.idx1]
    predicted = p0 @ B[:, :2].T + B[:, 2]
    err = np.linalg.norm(predicted - p1, axis=1)
    assert np.median(err) < 1.0  # full-resolution pixels, despite extracting at 800 px


def test_excluded_pixels_produce_no_keypoints():
    a, _, _ = view_pair()
    exclude = np.zeros(a.shape[:2], bool)
    exclude[200:600, 300:900] = True
    f = make_matching(OPENCV)[0].extract(a, exclude)
    inside = ((f.keypoints[:, 0] > 305) & (f.keypoints[:, 0] < 895)
              & (f.keypoints[:, 1] > 205) & (f.keypoints[:, 1] < 595))
    assert len(f) > 100 and not inside.any()


def test_unrelated_images_do_not_verify():
    ext, mat = make_matching(OPENCV)
    f0 = ext.extract(ground(1)[:800, :1200])
    f1 = ext.extract(ground(2)[:800, :1200])
    pair = verify_pair(f0, f1, mat.match(f0, f1), OPENCV)
    assert pair.F is None and len(pair.matches) == 0


def test_textureless_image_is_handled():
    ext, mat = make_matching(OPENCV)
    flat = ext.extract(np.full((400, 600, 3), 128, np.uint8))
    assert len(flat) == 0
    assert len(mat.match(flat, flat)) == 0


needs_lightglue = pytest.mark.skipif(not lightglue_available(), reason="lightglue not installed")
LIGHTGLUE = MatchingConfig(backend="lightglue", max_dim=800, max_keypoints=2048)


@needs_lightglue
def test_auto_picks_lightglue_when_installed():
    ext, mat = make_matching(MatchingConfig(backend="auto"))
    assert ext.name == "aliked+lightglue" and not ext.is_fallback and ext is mat


@needs_lightglue
def test_lightglue_matches_are_geometrically_correct():
    a, b, B = view_pair(angle=12.0)  # a larger rotation than the SIFT test
    ext, mat = make_matching(LIGHTGLUE)
    f0, f1 = ext.extract(a), ext.extract(b)
    pair = verify_pair(f0, f1, mat.match(f0, f1), LIGHTGLUE)
    p0, p1 = f0.keypoints[pair.matches.idx0], f1.keypoints[pair.matches.idx1]
    err = np.linalg.norm(p0 @ B[:, :2].T + B[:, 2] - p1, axis=1)
    assert len(pair.matches) > 300 and np.median(err) < 1.5  # full-resolution pixels
    assert pair.inlier_ratio > 0.8  # LightGlue proposes few wrong matches


@needs_lightglue
def test_lightglue_respects_the_exclusion_mask():
    a, _, _ = view_pair()
    exclude = np.zeros(a.shape[:2], bool)
    exclude[200:600, 300:900] = True
    f = make_matching(LIGHTGLUE)[0].extract(a, exclude)
    inside = ((f.keypoints[:, 0] > 305) & (f.keypoints[:, 0] < 895)
              & (f.keypoints[:, 1] > 205) & (f.keypoints[:, 1] < 595))
    assert len(f) > 100 and not inside.any()


def test_mask_shape_checked():
    with pytest.raises(DataError, match="mask shape"):
        make_matching(OPENCV)[0].extract(np.zeros((100, 100, 3), np.uint8), np.zeros((50, 50), bool))
