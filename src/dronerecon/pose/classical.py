"""Classical GPS-constrained reconstruction (the metric track).

Pairwise verified matches are joined into tracks, triangulated from initial
poses (the T0 preview placed in ENU), then refined by bundle adjustment that
minimises reprojection error together with GPS priors on camera centres
(weighted by the smoothed-track uncertainty). A weak prior on initial
orientation keeps roll about a straight flight line defined until IMU gravity
is applied in georeferencing.

Backend "scipy" is implemented here. "pycolmap" is reserved and reports that
the package is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from dronerecon.config import ClassicalConfig
from dronerecon.pose.geometry import STRAIGHT_TRACK_WARNING, is_near_straight, normalized_points
from dronerecon.pose.tracks import PairMatches, build_tracks, max_ray_angle_deg, triangulate
from dronerecon.pose.types import CameraPose, Trajectory
from dronerecon.types import DataError


@dataclass(frozen=True)
class SparseModel:
    points: np.ndarray  # Px3 ENU metres
    tracks: tuple[np.ndarray, ...]  # per point: Kx2 [keyframe position, keypoint index]
    point_error_px: np.ndarray  # P mean reprojection error


@dataclass(frozen=True)
class ReconstructionResult:
    trajectory: Trajectory
    sparse: SparseModel
    rms_px: float
    gps_rms_m: float  # camera centre vs GPS after adjustment
    warnings: tuple[str, ...] = ()


def _init_trajectory(initial: Trajectory, gps: np.ndarray) -> list[CameraPose]:
    """Valid initial poses as-is; invalid ones borrow the nearest valid rotation, GPS centre."""
    valid = np.flatnonzero(initial.valid)
    if valid.size == 0:
        raise DataError("classical: the initial trajectory has no valid poses")
    out = []
    for k, p in enumerate(initial.poses):
        if initial.valid[k]:
            out.append(p)
        else:
            near = valid[np.argmin(np.abs(valid - k))]
            out.append(CameraPose.from_center(initial.poses[near].R_cw, gps[k]))
    return out


def reconstruct(keypoints: Sequence[np.ndarray], pairs: Sequence[PairMatches], initial: Trajectory,
                gps_centers: np.ndarray, gps_sigma: np.ndarray,
                cfg: ClassicalConfig | None = None) -> ReconstructionResult:
    """keypoints: per keyframe Nx2 pixels. initial: T0 preview in ENU. gps_*: per keyframe Nx3."""
    cfg = cfg or ClassicalConfig()
    if cfg.backend == "pycolmap":
        raise DataError("pycolmap is not installed; set classical.backend to 'scipy'")
    if cfg.backend != "scipy":
        raise DataError(f"classical.backend must be 'scipy' or 'pycolmap', got '{cfg.backend}'")
    if initial.world != "enu":
        raise DataError("classical: initial trajectory must be placed in ENU (use the T0 preview)")
    n = len(initial)
    if not (len(keypoints) == n == len(gps_centers) == len(gps_sigma)):
        raise DataError("classical: keypoints, GPS and trajectory must have one entry per keyframe")

    K = initial.intrinsics
    norm = [normalized_points(kp.astype(float), K) if len(kp) else np.empty((0, 2)) for kp in keypoints]
    poses = _init_trajectory(initial, gps_centers)
    warnings: list[str] = []

    # ---- tracks + triangulation
    tracks = sorted(build_tracks([len(k) for k in keypoints], pairs), key=len, reverse=True)
    pts, kept = [], []
    for tr in tracks:
        cams = tr[:, 0]
        obs = np.array([norm[c][k] for c, k in tr])
        X = triangulate(poses, obs, cams)
        if not np.all(np.isfinite(X)):
            continue
        depths = [(poses[c].R_cw @ X + poses[c].t_cw)[2] for c in cams]
        if min(depths) <= 0 or max_ray_angle_deg(poses, cams, X) < cfg.min_tri_angle_deg:
            continue
        pts.append(X)
        kept.append(tr)
        if len(kept) >= cfg.max_points:
            break
    if len(kept) < 20:
        raise DataError(f"classical: only {len(kept)} points could be triangulated")

    points = np.array(pts)
    for _ in range(2):  # adjust, drop outlier observations, adjust again
        poses, points, err = _bundle_adjust(poses, points, kept, norm, K.fx, gps_centers, gps_sigma,
                                            initial, cfg)
        keep_obs = [e < cfg.max_reproj_px for e in err]
        new_tracks, new_pts = [], []
        for tr, X, ok in zip(kept, points, keep_obs):
            if ok.sum() >= 2:
                new_tracks.append(tr[ok])
                new_pts.append(X)
        dropped = sum(int((~ok).sum()) for ok in keep_obs)
        kept, points = new_tracks, np.array(new_pts)
        if dropped == 0:
            break
        warnings.append(f"removed {dropped} observation(s) with reprojection error above "
                        f"{cfg.max_reproj_px:g} px")

    _, _, err = _bundle_adjust(poses, points, kept, norm, K.fx, gps_centers, gps_sigma, initial, cfg,
                               evaluate_only=True)
    if is_near_straight(gps_centers, cfg.straight_track_ratio):
        warnings.append(STRAIGHT_TRACK_WARNING)
    all_err = np.concatenate(err)
    centers = np.array([p.center for p in poses])
    traj = Trajectory(initial.frame_indices, initial.t_video_s, tuple(poses), np.ones(n, bool), K,
                      "enu", True, "scipy-ba", False, tuple(warnings))
    return ReconstructionResult(
        traj, SparseModel(points, tuple(kept), np.array([e.mean() for e in err])),
        float(np.sqrt(np.mean(all_err ** 2))),
        float(np.sqrt(np.mean(np.sum((centers - gps_centers) ** 2, axis=1)))), tuple(warnings))


def _bundle_adjust(poses, points, tracks, norm, f, gps, gps_sigma, initial, cfg, evaluate_only=False):
    n, P = len(poses), len(points)
    cam_idx = np.concatenate([tr[:, 0] for tr in tracks])
    pt_idx = np.concatenate([np.full(len(tr), i) for i, tr in enumerate(tracks)])
    meas = np.concatenate([np.array([norm[c][k] for c, k in tr]) for tr in tracks])
    R0 = Rotation.from_matrix(np.array([p.R_cw for p in initial.poses]))
    rot_sigma = np.radians(cfg.rotation_prior_deg)

    def unpack(x):
        cams = x[: 6 * n].reshape(n, 6)
        return Rotation.from_rotvec(cams[:, :3]), cams[:, 3:], x[6 * n:].reshape(P, 3)

    def reproj(x):
        R, C, X = unpack(x)
        Rm = R.as_matrix()
        Xc = np.einsum("nij,nj->ni", Rm[cam_idx], X[pt_idx] - C[cam_idx])
        return (Xc[:, :2] / Xc[:, 2:3] - meas) * f, Xc[:, 2]

    fs = cfg.robust_scale_px

    def residuals(x):
        r, _ = reproj(x)
        R, C, _ = unpack(x)
        # Robust (soft-L1) weighting applies to image residuals only: rewrite each as
        # sign(r) * sqrt(rho(r)) so plain least squares minimises rho. GPS and rotation
        # priors stay quadratic; otherwise a far-off start lets the solution drift from GPS.
        r = r.ravel()
        r = np.sign(r) * np.sqrt(2 * fs ** 2 * (np.sqrt(1 + (r / fs) ** 2) - 1))
        gps_r = (C - gps) / gps_sigma
        # Gauge prior on the AVERAGE rotation only: it keeps roll about a straight flight line
        # defined without pinning each camera to a possibly drifting initial orientation.
        rot_r = (R * R0.inv()).as_rotvec().mean(0) / rot_sigma
        return np.concatenate([r, gps_r.ravel(), rot_r])

    x0 = np.concatenate([
        np.column_stack([Rotation.from_matrix(np.array([p.R_cw for p in poses])).as_rotvec(),
                         np.array([p.center for p in poses])]).ravel(),
        points.ravel()])

    if not evaluate_only:
        m = 2 * len(meas) + 3 * n + 3
        S = lil_matrix((m, len(x0)), dtype=int)
        rows = np.arange(len(meas))
        for d in range(2):
            for q in range(6):
                S[2 * rows + d, 6 * cam_idx + q] = 1
            for q in range(3):
                S[2 * rows + d, 6 * n + 3 * pt_idx + q] = 1
        base = 2 * len(meas)
        for c in range(n):
            S[base + 3 * c: base + 3 * c + 3, 6 * c + 3: 6 * c + 6] = 1  # GPS -> centre
            S[base + 3 * n: base + 3 * n + 3, 6 * c: 6 * c + 3] = 1  # mean-rotation prior -> every camera
        sol = least_squares(residuals, x0, jac_sparsity=S, loss="linear",
                            x_scale="jac", max_nfev=cfg.max_nfev, method="trf")
        x0 = sol.x

    R, C, X = unpack(x0)
    Rm = R.as_matrix()
    new_poses = [CameraPose.from_center(Rm[c], C[c]) for c in range(n)]
    r, _ = reproj(x0)
    per_obs = np.linalg.norm(r, axis=1)
    splits = np.cumsum([len(tr) for tr in tracks])[:-1]
    return new_poses, X, np.split(per_obs, splits)
