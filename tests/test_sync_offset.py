import cv2
import numpy as np
import pytest

from aerometric.config import SyncConfig
from aerometric.sync.offset import (
    EARTH_RADIUS_M,
    Signal,
    cross_correlate,
    estimate_rtk_offset,
    estimate_video_offset,
    flow_signal,
)
from aerometric.types import Frame, GpsTrack, ImuSeries

FPS = 10.0
TRUE_OFFSET = 0.7  # t_telemetry = t_video + offset


def speed_profile(t):
    """Ground speed in m/s with distinct accelerations so the correlation has a clear peak."""
    return 4 + 3 * np.sin(0.9 * t) + 2 * np.exp(-((t - 6) ** 2))


def _texture(seed=0, h=300, w=1200):
    rng = np.random.default_rng(seed)
    small = (rng.random((h // 3, w // 3)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), (3, 3), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_frames(n=100, px_per_m=10.0):
    """Camera translating east with the speed profile (sub-pixel accurate shifts)."""
    tex = _texture()
    t = np.arange(n) / FPS
    x = np.concatenate([[0], np.cumsum(speed_profile(t[:-1]) / FPS)]) * px_per_m
    frames = []
    for i in range(n):
        M = np.float32([[1, 0, -x[i]], [0, 1, -30]])
        frames.append(Frame(i, t[i], cv2.warpAffine(tex, M, (320, 240), flags=cv2.INTER_LINEAR)))
    return frames


def rotation_rate(t):
    return 0.3 + 0.2 * np.sin(0.6 * t) + 0.25 * np.exp(-((t - 4) ** 2))  # rad/s, non-periodic


def make_rotating_frames(n=100):
    """Camera hovering and yawing about the optical axis."""
    tex = _texture(1, 600, 600)
    t = np.arange(n) / FPS
    ang = np.concatenate([[0], np.cumsum(rotation_rate(t[:-1]) / FPS)])
    frames = []
    for i in range(n):
        M = cv2.getRotationMatrix2D((300, 300), np.degrees(ang[i]), 1.0)
        M[:, 2] -= (140, 180)
        frames.append(Frame(i, t[i], cv2.warpAffine(tex, M, (320, 240), flags=cv2.INTER_LINEAR)))
    return frames


def make_gps(offset=TRUE_OFFSET, duration=10.0, rate=10.0, lat0=28.6):
    t_video = np.arange(-1.0, duration + 1.0, 1 / rate)
    east = np.concatenate([[0], np.cumsum(speed_profile(t_video[:-1]) / rate)])
    lon = 77.2 + np.degrees(east / (EARTH_RADIUS_M * np.cos(np.radians(lat0))))
    n = len(t_video)
    return GpsTrack(t_video + offset, np.full(n, lat0), lon, np.full(n, 100.0))


def test_cross_correlate_recovers_shift():
    t = np.linspace(0, 20, 400)
    ref = Signal(t, np.sin(t) + 0.3 * np.sin(3.1 * t))
    other = Signal(t + 1.23, ref.v)
    peak = cross_correlate(ref, other)
    assert peak.lag_s == pytest.approx(1.23, abs=0.01)
    assert peak.correlation > 0.99


def test_periodic_signal_is_ambiguous():
    t = np.linspace(0, 20, 400)
    ref = Signal(t, np.sin(2 * np.pi * t / 2.0))  # 2 s period inside a ±5 s search
    peak = cross_correlate(ref, Signal(t + 0.3, ref.v))
    assert peak.is_ambiguous(SyncConfig())
    assert not peak.is_reliable(SyncConfig())


def test_flow_signal_tracks_speed():
    flow = flow_signal(make_frames())
    expected = speed_profile(flow.t) * 10.0  # px per metre
    assert np.corrcoef(flow.translation_px_s, expected)[0, 1] > 0.98
    assert np.nanmax(flow.rotation_rad_s) < 0.05


def test_video_offset_from_gps():
    est = estimate_video_offset(flow_signal(make_frames()), make_gps())
    assert est.method == "gps_speed"
    assert est.reliable
    assert est.offset_s == pytest.approx(TRUE_OFFSET, abs=0.05)


def test_constant_speed_is_flagged_unreliable():
    frames = make_frames()
    gps = make_gps()
    flat = GpsTrack(gps.t, gps.lat, np.linspace(gps.lon[0], gps.lon[-1], len(gps)), gps.alt_m)
    flow = flow_signal(frames)
    est = estimate_video_offset(flow, flat, cfg=SyncConfig())
    assert not est.reliable


def test_gyro_offset_for_hovering_yaw():
    flow = flow_signal(make_rotating_frames())
    assert np.corrcoef(flow.rotation_rad_s, rotation_rate(flow.t))[0, 1] > 0.95
    t_video = np.arange(-1, 11, 0.005)
    gyro = np.column_stack([np.zeros_like(t_video), np.zeros_like(t_video), rotation_rate(t_video)])
    imu = ImuSeries(t_video + TRUE_OFFSET, gyro)
    n = 50  # hovering: GPS barely moves, so its speed carries no timing information
    hover = GpsTrack(np.arange(n) * 0.25, np.full(n, 28.6), np.full(n, 77.2), np.full(n, 100.0))
    est = estimate_video_offset(flow, hover, imu)
    assert est.method == "gyro_rate" and est.reliable
    assert est.offset_s == pytest.approx(TRUE_OFFSET, abs=0.05)


def test_rtk_offset():
    tele = make_gps(offset=0.0)
    rtk = make_gps(offset=-0.4)  # rtk clock runs 0.4 s behind telemetry
    rtk = GpsTrack(rtk.t, rtk.lat, rtk.lon, rtk.alt_m, source="rtk")
    est = estimate_rtk_offset(tele, rtk)
    assert est.reliable
    assert est.offset_s == pytest.approx(0.4, abs=0.03)
