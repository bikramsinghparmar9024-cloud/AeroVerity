"""Robust similarity georeferencing (GPS + gravity).

Finds X_enu = s R X + t that best explains:
  * smoothed-GPS camera centres (robust loss; fixes beyond a threshold are outliers),
  * gravity observations (world up in camera frame), which pin down the tilt
    across a straight flight line that GPS alone cannot,
  * a weak prior toward the input orientation so the problem stays well posed.
The rotation uncertainty from the Jacobian tells us whether that tilt is
actually constrained, and we say so when it is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from dronerecon.config import GeorefConfig
from dronerecon.geo import LocalFrame
from dronerecon.georef.gravity import GravityObservations
from dronerecon.pose.geometry import umeyama
from dronerecon.pose.types import Similarity, Trajectory
from dronerecon.types import DataError


@dataclass(frozen=True)
class KnownDistance:
    """A real-world length between two points given in the input model's coordinates."""

    a: np.ndarray  # 3
    b: np.ndarray  # 3
    length_m: float
    sigma_m: float
    label: str = ""


@dataclass(frozen=True)
class GeoRegistration:
    transform: Similarity  # input world -> ENU
    trajectory: Trajectory  # metric, geo-registered poses
    points: np.ndarray  # sparse points in ENU
    frame: LocalFrame  # ENU origin (for latitude/longitude conversion)
    gps_rms_m: float  # inliers only
    gps_outliers: np.ndarray  # bool per keyframe
    gravity_rms_deg: float  # nan without gravity
    baro_rms_m: float  # nan without barometer
    known_residuals_m: np.ndarray  # model length - known length, per known distance
    rotation_std_deg: np.ndarray  # 1-sigma about east, north, up
    scale_std_rel: float  # 1-sigma relative scale uncertainty
    tilt_constrained: bool
    anchors: tuple[str, ...]  # what fixed the solution, e.g. ("gps", "gravity")
    warnings: tuple[str, ...] = ()


def _up_in_cam(traj: Trajectory, R: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """World up (0,0,1) seen from each camera after rotating the world by R."""
    Rcw = np.array([traj.poses[k].R_cw for k in idx])
    return np.einsum("nij,j->ni", Rcw @ R.T, [0.0, 0.0, 1.0])


def georeference(traj: Trajectory, sparse_points: np.ndarray, gps_centers: np.ndarray, gps_sigma: np.ndarray,
                 frame: LocalFrame, gravity: GravityObservations | None = None,
                 cfg: GeorefConfig | None = None, baro_alt_m: np.ndarray | None = None,
                 known: Sequence[KnownDistance] = ()) -> GeoRegistration:
    """baro_alt_m: barometric (relative) altitude per keyframe, NaN where missing. Its
    unknown zero offset is estimated; only altitude differences constrain the fit."""
    cfg = cfg or GeorefConfig()
    n = len(traj)
    if len(gps_centers) != n or len(gps_sigma) != n:
        raise DataError("georef: need one GPS centre and sigma per keyframe")
    if baro_alt_m is not None and len(baro_alt_m) != n:
        raise DataError("georef: need one barometer altitude per keyframe")
    ok = traj.valid & np.all(np.isfinite(gps_centers), axis=1)
    if ok.sum() < 3:
        raise DataError("georef: fewer than 3 keyframes with GPS")

    C = traj.centers
    init = umeyama(C[ok], gps_centers[ok])
    x0 = [np.log(init.s), *Rotation.from_matrix(init.R).as_rotvec(), *init.t]
    b_ok = traj.valid & np.isfinite(baro_alt_m) if baro_alt_m is not None else np.zeros(n, bool)
    use_baro = int(b_ok.sum()) >= 2
    if use_baro:
        x0.append(float(np.median(init.apply_points(C[b_ok])[:, 2] - baro_alt_m[b_ok])))
    x0 = np.array(x0)
    R_prior = Rotation.from_matrix(init.R)
    sig_rot = np.radians(cfg.rotation_prior_deg)
    g_idx = gravity.keyframes if gravity is not None else np.empty(0, int)
    g_obs = gravity.up_cam if gravity is not None else np.empty((0, 3))
    g_sig = np.radians(gravity.sigma_deg) if gravity is not None else np.empty(0)
    k_len = np.array([np.linalg.norm(np.asarray(k.a) - np.asarray(k.b)) for k in known])
    k_ref = np.array([k.length_m for k in known])
    k_sig = np.array([k.sigma_m for k in known])

    def unpack(x):
        return float(np.exp(x[0])), Rotation.from_rotvec(x[1:4]), x[4:7]

    use = ok.copy()

    def residuals(x):
        s, R, t = unpack(x)
        Rm = R.as_matrix()
        gps_r = ((s * C[use] @ Rm.T + t - gps_centers[use]) / gps_sigma[use]).ravel()
        rot_r = (R * R_prior.inv()).as_rotvec() / sig_rot
        parts = [gps_r]
        if len(g_idx):
            pred = _up_in_cam(traj, Rm, g_idx)
            parts.append((np.cross(pred, g_obs) / g_sig[:, None]).ravel())  # ~ angle / sigma
        if use_baro:
            z = s * C[b_ok] @ Rm[2] + t[2]
            parts.append((z - baro_alt_m[b_ok] - x[7]) / cfg.baro_sigma_m)
        if len(known):
            parts.append((s * k_len - k_ref) / k_sig)
        parts.append(rot_r)
        return np.concatenate(parts)

    def gps_outliers(x) -> np.ndarray:
        s, R, t = unpack(x)
        res = np.linalg.norm(s * C @ R.as_matrix().T + t - gps_centers, axis=1)
        return ok & (res / np.linalg.norm(gps_sigma, axis=1) * np.sqrt(3) > cfg.gps_outlier_sigmas)

    # robust fit, then refit without the flagged fixes so they no longer pull the solution
    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=cfg.gps_robust_sigmas, x_scale="jac")
    outliers = gps_outliers(sol.x)
    if outliers.any() and (ok & ~outliers).sum() >= 3:
        use = ok & ~outliers
        sol = least_squares(residuals, sol.x, loss="soft_l1", f_scale=cfg.gps_robust_sigmas, x_scale="jac")
        outliers = gps_outliers(sol.x)
    s, Rot, t = unpack(sol.x)
    R = Rot.as_matrix()
    sim = Similarity(s, R, t)

    # rotation uncertainty (world-frame axes) from the Jacobian
    J = sol.jac
    dof = max(1, len(sol.fun) - len(sol.x))
    s2 = max(1.0, float((sol.fun ** 2).sum()) / dof)
    cov = np.linalg.pinv(J.T @ J) * s2
    rot_std = np.degrees(np.sqrt(np.clip(np.diag(cov)[1:4], 0, None)))
    scale_std = float(np.sqrt(max(cov[0, 0], 0.0)))  # d(log s) = relative scale

    baro_rms = float("nan")
    if use_baro:
        z = s * C[b_ok] @ R[2] + t[2]
        baro_rms = float(np.sqrt(np.mean((z - baro_alt_m[b_ok] - sol.x[7]) ** 2)))
    known_res = s * k_len - k_ref if len(known) else np.empty(0)

    placed = traj.transformed(sim, "enu", True)
    res = np.linalg.norm(placed.centers - gps_centers, axis=1)
    inl = ok & ~outliers
    gps_rms = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else float("inf")

    g_rms = float("nan")
    if len(g_idx):
        pred = _up_in_cam(traj, R, g_idx)
        ang = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", pred, g_obs), -1, 1)))
        g_rms = float(np.sqrt(np.mean(ang ** 2)))

    # tilt across the flight: rotation about the horizontal axis along the track
    track = gps_centers[ok] - gps_centers[ok].mean(0)
    along = np.linalg.svd(track, full_matrices=False)[2][0]
    along[2] = 0.0
    along = along / (np.linalg.norm(along) or 1.0)
    roll_std = float(np.sqrt(along @ (np.diag(rot_std) ** 2) @ along))
    constrained = roll_std <= cfg.max_roll_std_deg

    anchors = (["gps"] + (["gravity"] if len(g_idx) else []) + (["barometer"] if use_baro else [])
               + (["known_dimensions"] if len(known) else []))
    warnings: list[str] = []
    if outliers.any():
        warnings.append(f"{int(outliers.sum())} keyframe GPS position(s) disagreed with the model and "
                        "were left out of the fit")
    if not constrained:
        warnings.append(f"the model's tilt across the flight line is uncertain (±{roll_std:.1f}°); "
                        "gimbal attitude or IMU data would fix it")
    if len(g_idx) and g_rms > 3 * float(np.median(gravity.sigma_deg)):
        warnings.append(f"gravity from the gimbal disagrees with the model by {g_rms:.1f}° RMS; "
                        "check the gimbal roll sign in the configuration")
    for k, r in zip(known, known_res):
        if abs(r) > 3 * k.sigma_m:
            name = f"'{k.label}'" if k.label else "a known dimension"
            warnings.append(f"{name} measures {k.length_m + r:.2f} m in the model but should be "
                            f"{k.length_m:.2f} m; check the reference points or the value")
    return GeoRegistration(sim, placed, sim.apply_points(sparse_points) if len(sparse_points) else sparse_points,
                           frame, gps_rms, outliers, g_rms, baro_rms, known_res, rot_std, scale_std, constrained,
                           tuple(anchors), tuple(warnings))
