import cv2
import numpy as np
import pytest

from dronerecon.config import MatchingConfig, PairingConfig
from dronerecon.matching.features import make_matching
from dronerecon.matching.pairs import VladRetrieval, global_descriptors, select_pairs
from dronerecon.types import DataError


def as_set(pairs, reason=None):
    return {(p.i, p.j) for p in pairs if reason is None or reason in p.reasons}


def test_sequence_window():
    pairs = select_pairs(6, PairingConfig(sequence_window=2))
    assert as_set(pairs) == {(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)}
    assert all(p.reasons == {"sequence"} for p in pairs)


def test_gps_pairs_link_adjacent_flight_lines():
    # Two parallel lines 20 m apart, flown out and back (a lawnmower turn).
    x = np.arange(10) * 15.0
    pos = np.concatenate([np.column_stack([x, np.zeros(10), np.zeros(10)]),
                          np.column_stack([x[::-1], np.full(10, 20.0), np.zeros(10)])])
    pairs = select_pairs(20, PairingConfig(sequence_window=2, gps_radius_m=25.0), positions=pos)
    gps = as_set(pairs, "gps")
    assert (0, 19) in gps and (9, 10) in gps  # same x on the two lines
    assert (0, 10) not in gps  # 135 m apart
    both = [p for p in pairs if p.reasons == {"sequence", "gps"}]
    assert both  # reasons are merged, not duplicated


def ground(w=2400, h=700, seed=3):
    rng = np.random.default_rng(seed)
    img = cv2.resize((rng.random((h // 8, w // 8, 3)) * 255).astype(np.uint8), (w, h),
                     interpolation=cv2.INTER_CUBIC)
    for _ in range(250):
        x, y = rng.integers(0, w - 50), rng.integers(0, h - 50)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(5, 25)),
                   tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return img


@pytest.fixture(scope="module")
def revisit_features():
    """Pass 1 over x = 0..1800, then a return pass over the same ground (loop)."""
    g = ground()
    xs = list(range(0, 1900, 200))  # 10 views
    views = [g[50:650, x:x + 600] for x in xs] + [g[60:660, x + 10:x + 610] for x in xs[::-1]]
    ext, _ = make_matching(MatchingConfig(backend="opencv", max_dim=600, max_keypoints=1500))
    return [ext.extract(np.ascontiguousarray(v)) for v in views]


def test_vlad_retrieval_finds_revisits(revisit_features):
    desc = global_descriptors(revisit_features, PairingConfig(vlad_clusters=16))
    assert desc.shape[0] == 20
    assert np.allclose(np.linalg.norm(desc, axis=1), 1.0, atol=1e-5)
    pairs = select_pairs(20, PairingConfig(sequence_window=2, retrieval_top_k=2), descriptors=desc)
    loops = as_set(pairs, "retrieval")
    expected = {(i, 19 - i) for i in range(10) if abs(19 - 2 * i) > 2}
    assert len(expected & loops) >= 0.8 * len(expected)
    assert all(np.isfinite(p.similarity) for p in pairs if "retrieval" in p.reasons)


def test_gps_gate_blocks_implausible_loops(revisit_features):
    desc = global_descriptors(revisit_features, PairingConfig(vlad_clusters=16))
    far = np.column_stack([np.arange(20) * 500.0, np.zeros(20), np.zeros(20)])  # all > 150 m apart
    pairs = select_pairs(20, PairingConfig(sequence_window=2), positions=far, descriptors=desc)
    assert not as_set(pairs, "retrieval") and not as_set(pairs, "gps")


def test_deterministic(revisit_features):
    a = global_descriptors(revisit_features, PairingConfig(vlad_clusters=16), seed=7)
    b = global_descriptors(revisit_features, PairingConfig(vlad_clusters=16), seed=7)
    assert np.array_equal(a, b)


def test_input_checks():
    with pytest.raises(DataError, match="positions"):
        select_pairs(3, positions=np.zeros((2, 3)))
    with pytest.raises(DataError, match="fit"):
        VladRetrieval().encode(None)
