"""Video ↔ telemetry time-offset estimation.

Motion seen in the video (optical flow) is cross-correlated with motion
reported by telemetry (GPS ground speed, gyro rate). Convention everywhere:

    t_telemetry = t_video + offset_s

The same correlator aligns an RTK/PPK track to the telemetry clock using
ground speed from both tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import cv2
import numpy as np

from aerometric.config import SyncConfig
from aerometric.types import DataError, Frame, GpsTrack, ImuSeries

EARTH_RADIUS_M = 6_378_137.0
SyncMethod = Literal["gps_speed", "gyro_rate", "none"]


@dataclass(frozen=True)
class Signal:
    t: np.ndarray
    v: np.ndarray


@dataclass(frozen=True)
class FlowSignal:
    """Per frame-pair image motion, timed at the pair midpoint (video clock)."""

    t: np.ndarray
    translation_px_s: np.ndarray  # magnitude of image translation
    rotation_rad_s: np.ndarray  # in-plane rotation rate magnitude


@dataclass(frozen=True)
class OffsetEstimate:
    offset_s: float
    correlation: float  # peak normalised cross-correlation, -1..1
    reliable: bool
    method: SyncMethod


# --------------------------------------------------------------------------- signals

def flow_signal(frames: Iterable[Frame], cfg: SyncConfig | None = None) -> FlowSignal:
    """Estimate frame-to-frame image motion with sparse LK flow + a robust similarity fit."""
    cfg = cfg or SyncConfig()
    t_out, trans, rot = [], [], []
    prev_gray, prev_t, scale = None, 0.0, 1.0
    for f in frames:
        h, w = f.image.shape[:2]
        scale = min(1.0, cfg.flow_width / w)
        small = cv2.resize(f.image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        if prev_gray is not None:
            dt = f.t_video_s - prev_t
            tx, ty, ang = _pair_motion(prev_gray, gray, cfg.max_corners)
            t_out.append((f.t_video_s + prev_t) / 2)
            trans.append(np.hypot(tx, ty) / scale / dt)
            rot.append(abs(ang) / dt)
        prev_gray, prev_t = gray, f.t_video_s
    if len(t_out) < 2:
        raise DataError("flow_signal: need at least 3 frames")
    return FlowSignal(np.array(t_out), np.array(trans), np.array(rot))


def _pair_motion(a: np.ndarray, b: np.ndarray, max_corners: int) -> tuple[float, float, float]:
    pts = cv2.goodFeaturesToTrack(a, max_corners, 0.01, 7)
    if pts is None or len(pts) < 8:
        return np.nan, np.nan, np.nan
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
    ok = status.ravel() == 1
    if ok.sum() < 8:
        return np.nan, np.nan, np.nan
    M, _ = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC)
    if M is None:
        return np.nan, np.nan, np.nan
    return float(M[0, 2]), float(M[1, 2]), float(np.arctan2(M[1, 0], M[0, 0]))


def local_en(track: GpsTrack) -> tuple[np.ndarray, np.ndarray]:
    """East/north metres relative to the first sample (equirectangular; fine for < ~10 km)."""
    lat0 = np.radians(track.lat[0])
    east = np.radians(track.lon - track.lon[0]) * EARTH_RADIUS_M * np.cos(lat0)
    north = np.radians(track.lat - track.lat[0]) * EARTH_RADIUS_M
    return east, north


def gps_speed_signal(track: GpsTrack, window_s: float = 0.5) -> Signal:
    """Ground speed measured across `window_s` rather than between adjacent fixes.

    Differencing fixes logged at 10-30 Hz turns centimetre-level position jitter into
    metres-per-second of speed noise; a half-second baseline keeps the real changes.
    """
    east, north = local_en(track)
    t = track.t
    if window_s <= 0 or t[-1] - t[0] <= window_s:
        dt = np.diff(t)
        return Signal((t[1:] + t[:-1]) / 2, np.hypot(np.diff(east), np.diff(north)) / dt)
    tc = t[(t >= t[0] + window_s / 2) & (t <= t[-1] - window_s / 2)]
    e0, e1 = np.interp(tc - window_s / 2, t, east), np.interp(tc + window_s / 2, t, east)
    n0, n1 = np.interp(tc - window_s / 2, t, north), np.interp(tc + window_s / 2, t, north)
    return Signal(tc, np.hypot(e1 - e0, n1 - n0) / window_s)


def gyro_rate_signal(imu: ImuSeries) -> Signal:
    return Signal(imu.t, np.linalg.norm(imu.gyro_rad_s, axis=1))


# --------------------------------------------------------------------------- correlation

MIN_RELATIVE_STD = 1e-4  # below this a signal carries no timing information


def _finite(sig: Signal) -> Signal:
    ok = np.isfinite(sig.v)
    return Signal(sig.t[ok], sig.v[ok])


def _informative(sig: Signal) -> bool:
    scale = max(float(np.abs(sig.v).mean()), 1e-12)
    return float(sig.v.std()) > MIN_RELATIVE_STD * scale


@dataclass(frozen=True)
class Peak:
    lag_s: float
    correlation: float
    runner_up: float  # best correlation at least peak_exclusion_s away from lag_s

    def is_reliable(self, cfg: SyncConfig) -> bool:
        return (np.isfinite(self.correlation) and self.correlation >= cfg.min_correlation
                and not self.is_ambiguous(cfg))

    def is_ambiguous(self, cfg: SyncConfig) -> bool:
        return bool(np.isfinite(self.runner_up)
                    and self.runner_up >= self.correlation - cfg.ambiguity_margin)


_NO_PEAK = Peak(0.0, float("nan"), float("nan"))


def cross_correlate(ref: Signal, other: Signal, cfg: SyncConfig | None = None) -> Peak:
    """Find lag L maximising corr(ref(t), other(t + L))."""
    cfg = cfg or SyncConfig()
    ref, other = _finite(ref), _finite(other)
    if len(ref.t) < 4 or len(other.t) < 4:
        raise DataError("cross_correlate: signals too short")
    if not (_informative(ref) and _informative(other)):
        return _NO_PEAK  # e.g. perfectly constant speed: no peak to find
    grid = np.arange(ref.t[0], ref.t[-1], cfg.grid_dt_s)
    a = np.interp(grid, ref.t, ref.v)
    shorter = min(ref.t[-1] - ref.t[0], other.t[-1] - other.t[0])
    min_n = max(4, int(cfg.min_overlap * shorter / cfg.grid_dt_s))

    lags = np.arange(-cfg.max_lag_s, cfg.max_lag_s + cfg.grid_dt_s / 2, cfg.grid_dt_s)
    scores = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        tb = grid + lag
        ok = (tb >= other.t[0]) & (tb <= other.t[-1])
        if ok.sum() < min_n:
            continue
        x, y = a[ok], np.interp(tb[ok], other.t, other.v)
        sx, sy = x.std(), y.std()
        if sx > 0 and sy > 0:
            scores[i] = float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))
    if np.all(np.isnan(scores)):
        return _NO_PEAK
    k = int(np.nanargmax(scores))
    lag = float(lags[k])
    if 0 < k < len(lags) - 1 and np.isfinite(scores[k - 1]) and np.isfinite(scores[k + 1]):
        y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
        denom = y0 - 2 * y1 + y2
        if denom < 0:  # parabolic sub-sample refinement
            lag += 0.5 * (y0 - y2) / denom * cfg.grid_dt_s
    far = np.abs(lags - lags[k]) >= cfg.peak_exclusion_s
    rival = scores[far & np.isfinite(scores)]
    return Peak(lag, float(scores[k]), float(rival.max()) if rival.size else float("nan"))


# --------------------------------------------------------------------------- public API

def estimate_video_offset(flow: FlowSignal, gps: GpsTrack, imu: ImuSeries | None = None,
                          cfg: SyncConfig | None = None) -> OffsetEstimate:
    """Pick the telemetry signal (GPS speed or gyro rate) that gives the most trustworthy peak."""
    cfg = cfg or SyncConfig()
    candidates: list[tuple[SyncMethod, Peak]] = [
        ("gps_speed", cross_correlate(Signal(flow.t, flow.translation_px_s), gps_speed_signal(gps), cfg))
    ]
    if imu is not None:
        # Gimbal-stabilised cameras see little body rotation, so this often loses to GPS.
        candidates.append(("gyro_rate", cross_correlate(Signal(flow.t, flow.rotation_rad_s),
                                                        gyro_rate_signal(imu), cfg)))
    usable = [c for c in candidates if np.isfinite(c[1].correlation)]
    if not usable:
        return OffsetEstimate(0.0, float("nan"), False, "none")
    method, peak = max(usable, key=lambda c: (c[1].is_reliable(cfg), c[1].correlation))
    return OffsetEstimate(peak.lag_s, peak.correlation, peak.is_reliable(cfg), method)


def estimate_rtk_offset(telemetry_gps: GpsTrack, rtk: GpsTrack,
                        cfg: SyncConfig | None = None) -> OffsetEstimate:
    """Offset such that t_telemetry = t_rtk + offset_s, from ground-speed correlation.

    Assumes the RTK track's zero is already roughly aligned (within max_lag_s),
    e.g. by matching wall-clock start times.
    """
    cfg = cfg or SyncConfig()
    peak = cross_correlate(gps_speed_signal(rtk), gps_speed_signal(telemetry_gps), cfg)
    return OffsetEstimate(peak.lag_s, peak.correlation, peak.is_reliable(cfg), "gps_speed")
