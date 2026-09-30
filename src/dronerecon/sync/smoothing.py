"""Kalman filter + Rauch–Tung–Striebel smoother for the GPS track.

State per sample: ENU position and velocity (constant-velocity model driven by
white-noise acceleration). Fixes whose innovation fails a chi-square gate are
treated as outliers and skipped. The smoothed track stays on the telemetry
clock; use `SmoothedTrack.interpolate(t_video + offset_s)` to query per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dronerecon.config import SmoothingConfig
from dronerecon.geo import LocalFrame
from dronerecon.sync.offset import OffsetEstimate
from dronerecon.types import DataError, FlightRecord, GpsSource, GpsTrack

_I3 = np.eye(3)


@dataclass(frozen=True)
class SmoothedTrack:
    t: np.ndarray  # telemetry clock, seconds
    enu: np.ndarray  # Nx3 metres in `frame`
    vel: np.ndarray  # Nx3 m/s
    pos_std: np.ndarray  # Nx3 metres, 1-sigma from the smoother covariance
    frame: LocalFrame
    source: GpsSource
    outlier: np.ndarray  # bool per input fix used, True = rejected by the gate
    warnings: tuple[str, ...] = ()

    def interpolate(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ENU position and 1-sigma at query times (clamped to the track span)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        pos = np.column_stack([np.interp(t, self.t, self.enu[:, i]) for i in range(3)])
        std = np.column_stack([np.interp(t, self.t, self.pos_std[:, i]) for i in range(3)])
        return pos, std

    def to_gps_track(self) -> GpsTrack:
        lat, lon, alt = self.frame.to_lla(self.enu)
        return GpsTrack(self.t, lat, lon, alt, source=self.source,
                        h_acc_m=np.hypot(self.pos_std[:, 0], self.pos_std[:, 1]),
                        v_acc_m=self.pos_std[:, 2])


# --------------------------------------------------------------------------- measurement selection

def select_measurements(record: FlightRecord,
                        rtk_offset: OffsetEstimate | None) -> tuple[GpsTrack, list[str]]:
    """Use the RTK/PPK track (shifted onto the telemetry clock) when it can be trusted."""
    if record.rtk is None:
        return record.gps, []
    if rtk_offset is None or not rtk_offset.reliable:
        return record.gps, ["RTK/PPK track ignored: its clock could not be aligned with telemetry"]
    r = record.rtk
    shifted = GpsTrack(r.t + rtk_offset.offset_s, r.lat, r.lon, r.alt_m, r.source, r.h_acc_m, r.v_acc_m)
    return shifted, []


def _drop_stale(track: GpsTrack) -> np.ndarray:
    """Keep the first of each run of identical consecutive fixes."""
    same = (np.diff(track.lat) == 0) & (np.diff(track.lon) == 0) & (np.diff(track.alt_m) == 0)
    return np.concatenate([[True], ~same])


def _measurement_sigma(track: GpsTrack, keep: np.ndarray, cfg: SmoothingConfig) -> np.ndarray:
    n = int(keep.sum())
    sh = (track.h_acc_m[keep] / np.sqrt(2) if track.h_acc_m is not None
          else np.full(n, cfg.gnss_sigma_h_m))
    sv = track.v_acc_m[keep] if track.v_acc_m is not None else np.full(n, cfg.gnss_sigma_v_m)
    return np.column_stack([sh, sh, sv]).clip(min=0.01)


# --------------------------------------------------------------------------- filter

def _transition(dt: float, cfg: SmoothingConfig) -> tuple[np.ndarray, np.ndarray]:
    F = np.eye(6)
    F[:3, 3:] = dt * _I3
    q = np.diag([cfg.accel_noise_h, cfg.accel_noise_h, cfg.accel_noise_v]) ** 2
    Q = np.block([[dt ** 3 / 3 * q, dt ** 2 / 2 * q], [dt ** 2 / 2 * q, dt * q]])
    return F, Q


def smooth_gps(track: GpsTrack, cfg: SmoothingConfig | None = None) -> SmoothedTrack:
    cfg = cfg or SmoothingConfig()
    keep = _drop_stale(track) if cfg.drop_stale else np.ones(len(track), bool)
    if keep.sum() < 3:
        raise DataError("smooth_gps: fewer than 3 distinct GPS fixes")
    warnings = []
    if (stale := int((~keep).sum())):
        warnings.append(f"ignored {stale} repeated GPS fix(es) logged between receiver updates")

    t = track.t[keep]
    frame = LocalFrame(float(track.lat[keep][0]), float(track.lon[keep][0]), float(track.alt_m[keep][0]))
    z = frame.to_enu(track.lat[keep], track.lon[keep], track.alt_m[keep])
    sig = _measurement_sigma(track, keep, cfg)
    n = len(t)

    xf, Pf = np.zeros((n, 6)), np.zeros((n, 6, 6))
    xp, Pp = np.zeros((n, 6)), np.zeros((n, 6, 6))
    Fs = np.zeros((n, 6, 6))
    outlier = np.zeros(n, bool)

    x = np.concatenate([z[0], np.zeros(3)])
    P = np.diag(np.concatenate([sig[0] ** 2, np.full(3, 25.0)]))  # velocity unknown: 5 m/s sigma
    for k in range(n):
        if k > 0:
            F, Q = _transition(float(t[k] - t[k - 1]), cfg)
            x, P = F @ x, F @ P @ F.T + Q
            Fs[k] = F
        xp[k], Pp[k] = x, P
        R = np.diag(sig[k] ** 2)
        y = z[k] - x[:3]
        S = P[:3, :3] + R
        if k > 0 and float(y @ np.linalg.solve(S, y)) > cfg.gate_chi2:
            outlier[k] = True
        else:
            K = P[:, :3] @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(6) - K[:, :3] @ np.eye(3, 6)) @ P
            P = (P + P.T) / 2
        xf[k], Pf[k] = x, P

    xs, Ps = xf.copy(), Pf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ Fs[k + 1].T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T

    if outlier.any():
        warnings.append(f"rejected {int(outlier.sum())} GPS fix(es) as outliers")
    std = np.sqrt(np.clip(np.diagonal(Ps, axis1=1, axis2=2)[:, :3], 0, None))
    return SmoothedTrack(t, xs[:, :3], xs[:, 3:], std, frame, track.source, outlier, tuple(warnings))


def smooth_flight(record: FlightRecord, rtk_offset: OffsetEstimate | None = None,
                  cfg: SmoothingConfig | None = None) -> SmoothedTrack:
    """Smoothed GPS track for a flight, preferring aligned RTK/PPK when available."""
    track, warnings = select_measurements(record, rtk_offset)
    result = smooth_gps(track, cfg)
    if warnings:
        result = SmoothedTrack(result.t, result.enu, result.vel, result.pos_std, result.frame,
                               result.source, result.outlier, tuple(warnings) + result.warnings)
    return result
