import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from aerometric.calibration import (
    PairCorrespondences,
    assess_rolling_shutter,
    distort_normalized,
    self_calibrate,
    undistort_normalized,
)
from aerometric.config import CalibrationConfig
from aerometric.sync.offset import FlowSignal
from aerometric.types import CameraIntrinsics

W, H = 1920, 1080
F_TRUE, K1, K2 = 1450.0, -0.12, 0.03


def cameras(n=8, alt=100.0, tilt_deg=25.0):
    """Oblique cameras along a line with varying yaw/pitch (looking down and forward)."""
    out = []
    for i in range(n):
        C = np.array([i * 12.0, 0.0, alt])
        R_wc = Rotation.from_euler("xyz", [180 - tilt_deg + 4 * np.sin(i), 3 * np.cos(i), 6 * i],
                                   degrees=True).as_matrix()
        out.append((R_wc.T, C))
    return out


def project(R, C, X, f=F_TRUE, k1=K1, k2=K2):
    Xc = (X - C) @ R.T
    ok = Xc[:, 2] > 1
    xn = Xc[:, :2] / Xc[:, 2:3]
    uv = distort_normalized(xn, k1, k2) * f + [W / 2, H / 2]
    ok &= (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)
    return uv, ok


def scene(relief=25.0, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-80, 180, n), rng.uniform(-150, 150, n),
                         rng.uniform(0, relief, n)])
    return X


def make_pairs(X, cams, noise=0.4, seed=1):
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(len(cams)):
        for j in range(i + 1, min(len(cams), i + 3)):
            a, oka = project(*cams[i], X)
            b, okb = project(*cams[j], X)
            ok = oka & okb
            pairs.append(PairCorrespondences(a[ok] + rng.normal(0, noise, (ok.sum(), 2)),
                                             b[ok] + rng.normal(0, noise, (ok.sum(), 2))))
    return pairs


def test_distortion_roundtrip():
    x = np.random.default_rng(0).uniform(-0.7, 0.7, (100, 2))
    assert undistort_normalized(distort_normalized(x, K1, K2), K1, K2) == pytest.approx(x, abs=1e-6)


def test_recovers_focal_and_distortion_from_a_20_percent_wrong_prior():
    prior = CameraIntrinsics(W, H, F_TRUE * 1.2, F_TRUE * 1.2, W / 2, H / 2, source="default")
    res = self_calibrate(make_pairs(scene(), cameras()), prior)
    assert res.observable, res.warnings
    assert res.intrinsics.source == "estimated"
    assert res.intrinsics.fx == pytest.approx(F_TRUE, rel=0.02)
    assert res.intrinsics.dist[0] == pytest.approx(K1, abs=0.03)
    assert res.focal_rel_std < 0.05 and res.rms_px < 1.0


def test_flat_ground_straight_flight_is_unobservable():
    cams = [(Rotation.from_euler("x", 180, degrees=True).as_matrix().T, np.array([i * 12.0, 0, 100.0]))
            for i in range(8)]  # pure nadir translation, no rotation
    prior = CameraIntrinsics(W, H, F_TRUE * 1.2, F_TRUE * 1.2, W / 2, H / 2, source="default")
    res = self_calibrate(make_pairs(scene(relief=0.0), cams), prior)
    assert not res.observable
    assert res.intrinsics is prior
    assert "could not be pinned down" in res.warnings[0]


def test_provided_intrinsics_are_kept_by_default():
    k = CameraIntrinsics(W, H, 1400, 1400, W / 2, H / 2, source="provided")
    res = self_calibrate([], k)
    assert res.intrinsics is k and res.observable


def test_rolling_shutter_assessment():
    fast = FlowSignal(np.arange(10.0), np.full(10, 400.0), np.zeros(10))  # 400 px/s
    unknown = assess_rolling_shutter(fast)
    assert not unknown.applicable and "Set calibration.readout_s" in unknown.message
    known = assess_rolling_shutter(fast, CalibrationConfig(readout_s=0.025))
    assert known.applicable and known.p95_skew_px == pytest.approx(10.0)
    slow = assess_rolling_shutter(FlowSignal(np.arange(10.0), np.full(10, 20.0), np.zeros(10)),
                                  CalibrationConfig(readout_s=0.025))
    assert not slow.applicable and "global shutter" in slow.message
