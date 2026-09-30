import numpy as np
import pytest

from aerometric.geo import LocalFrame
from aerometric.sync.offset import OffsetEstimate
from aerometric.sync.smoothing import smooth_flight, smooth_gps
from aerometric.types import CameraIntrinsics, FlightRecord, GpsTrack, VideoInfo

FRAME = LocalFrame(28.6, 77.2, 100.0)


def truth(t):
    """Curving pass: ~8 m/s east with a gentle north turn and a climb."""
    return np.column_stack([8 * t, 0.05 * t ** 2, 100 + 0.5 * t])


def noisy_track(seed=0, rate=5.0, duration=60.0, sigma_h=1.5, sigma_v=3.0, outliers=()):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / rate)
    enu = truth(t) + rng.normal(0, [sigma_h, sigma_h, sigma_v], (len(t), 3))
    for i in outliers:
        enu[i] += [40.0, -35.0, 0.0]
    enu[:, 2] -= FRAME.alt0  # truth() gives absolute altitude
    return t, GpsTrack(t, *FRAME.to_lla(enu))


def truth_enu(t):
    return truth(t) - [0, 0, FRAME.alt0]


def _errors(sm):
    """Smoothed minus true positions, both in the test's fixed frame."""
    return FRAME.to_enu(*sm.frame.to_lla(sm.enu)) - truth_enu(sm.t)


def test_geo_roundtrip():
    enu = np.array([[120.0, -340.0, 12.0]])
    back = FRAME.to_enu(*FRAME.to_lla(enu))
    assert back == pytest.approx(enu, abs=1e-6)


def test_smoothing_reduces_error_and_reports_honest_sigma():
    t, track = noisy_track()
    raw = FRAME.to_enu(track.lat, track.lon, track.alt_m) - truth_enu(t)
    sm = smooth_gps(track)
    err = _errors(sm)
    rms_raw = np.sqrt((raw[:, :2] ** 2).mean())
    rms_sm = np.sqrt((err[:, :2] ** 2).mean())
    assert rms_sm < 0.5 * rms_raw
    inside = np.abs(err) < 3 * sm.pos_std
    assert inside.mean() > 0.95  # reported sigma is not over-confident
    true_vel = np.column_stack([np.full_like(sm.t, 8.0), 0.1 * sm.t, np.full_like(sm.t, 0.5)])
    assert np.median(np.linalg.norm(sm.vel - true_vel, axis=1)) < 1.0


def test_outliers_rejected():
    t, track = noisy_track(outliers=(50, 120, 121))
    sm = smooth_gps(track)
    assert set(np.flatnonzero(sm.outlier)) >= {50, 120, 121}
    assert sm.outlier.sum() <= 6
    assert np.abs(_errors(sm)[:, :2]).max() < 5.0
    assert any("outlier" in w for w in sm.warnings)


def test_stale_repeats_dropped():
    t, track = noisy_track(duration=10.0)
    rep = np.repeat(np.arange(len(t)), 3)  # logged at 3x the receiver rate
    t_fast = np.arange(len(rep)) / 15.0
    fast = GpsTrack(t_fast, track.lat[rep], track.lon[rep], track.alt_m[rep])
    sm = smooth_gps(fast)
    assert len(sm.t) == len(t)
    assert any("repeated" in w for w in sm.warnings)


def test_interpolate_and_export():
    _, track = noisy_track(duration=20.0)
    sm = smooth_gps(track)
    pos, std = sm.interpolate([5.1, 100.0])
    assert pos.shape == (2, 3) and np.all(std > 0)
    assert pos[1] == pytest.approx(sm.enu[-1])  # clamped at the end
    g = sm.to_gps_track()
    assert len(g) == len(sm.t) and g.h_acc_m is not None


def _record(track, rtk=None):
    return FlightRecord(VideoInfo("v.mp4", 64, 48, 10.0, 100), track,
                        CameraIntrinsics.from_fov(64, 48, 80), rtk=rtk)


def test_rtk_preferred_only_when_aligned():
    _, gnss = noisy_track(seed=1, duration=20.0)
    t, precise = noisy_track(seed=2, duration=20.0, sigma_h=0.03, sigma_v=0.05)
    rtk = GpsTrack(t - 0.4, precise.lat, precise.lon, precise.alt_m, source="rtk",
                   h_acc_m=np.full(len(t), 0.04), v_acc_m=np.full(len(t), 0.05))
    good = smooth_flight(_record(gnss, rtk), OffsetEstimate(0.4, 0.95, True, "gps_speed"))
    assert good.source == "rtk"
    assert good.t[0] == pytest.approx(0.0)
    assert np.median(good.pos_std[:, 0]) < 0.1

    bad = smooth_flight(_record(gnss, rtk), OffsetEstimate(0.0, 0.2, False, "gps_speed"))
    assert bad.source == "gnss"
    assert any("RTK/PPK track ignored" in w for w in bad.warnings)
