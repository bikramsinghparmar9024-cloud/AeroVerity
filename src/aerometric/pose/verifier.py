"""Pose verifier: accept the AI trajectory only if its residuals pass.

Both trajectories are scored the same way against the same evidence:
  * reprojection: every sparse track is re-triangulated with the trajectory's
    own poses and reprojected into each observing keyframe;
  * GPS: camera centres vs smoothed GPS, normalised by the GPS uncertainty.
The AI track is accepted when it passes all thresholds; otherwise the
classical (bundle-adjusted) track is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from aerometric.config import VerifierConfig
from aerometric.pose.classical import ReconstructionResult
from aerometric.pose.geometry import normalized_points
from aerometric.pose.tracks import triangulate
from aerometric.pose.types import Trajectory
from aerometric.types import DataError

Choice = Literal["ai", "classical"]
BEHIND_CAMERA_PX = 1e6  # reprojection error recorded for a point that ends up behind the camera


@dataclass(frozen=True)
class TrajectoryMetrics:
    reproj_median_px: float
    reproj_p90_px: float
    gps_rms_m: float
    gps_nrms: float  # RMS of residual / sigma
    valid_fraction: float
    tracks_used: int


@dataclass(frozen=True)
class VerificationResult:
    accepted: Choice
    trajectory: Trajectory
    ai: TrajectoryMetrics
    classical: TrajectoryMetrics
    reasons: tuple[str, ...]  # plain-language explanation of the decision


def evaluate_trajectory(traj: Trajectory, keypoints: Sequence[np.ndarray], tracks: Sequence[np.ndarray],
                        gps_centers: np.ndarray, gps_sigma: np.ndarray,
                        cfg: VerifierConfig | None = None) -> TrajectoryMetrics:
    cfg = cfg or VerifierConfig()
    if len(keypoints) != len(traj):
        raise DataError("verifier: need one keypoint array per keyframe")
    K = traj.intrinsics
    norm = [normalized_points(k.astype(float), K) if len(k) else np.empty((0, 2)) for k in keypoints]
    step = max(1, len(tracks) // cfg.max_tracks)
    errs = []
    used = 0
    for tr in tracks[::step]:
        cams = tr[:, 0]
        if not traj.valid[cams].all():
            continue
        obs = np.array([norm[c][k] for c, k in tr])
        X = triangulate(traj.poses, obs, cams)
        if not np.all(np.isfinite(X)):
            continue
        used += 1
        for (c, _), o in zip(tr, obs):
            p = traj.poses[c]
            Xc = p.R_cw @ X + p.t_cw
            if Xc[2] <= 0:
                errs.append(np.inf)  # point behind the camera: pose is inconsistent
            else:
                errs.append(float(np.linalg.norm(Xc[:2] / Xc[2] - o) * K.fx))
    # Points behind a camera count as a huge (finite) miss: np.percentile turns inf into NaN,
    # and NaN would silently pass every "error > limit" check.
    e = np.minimum(np.array(errs), BEHIND_CAMERA_PX) if errs else np.array([BEHIND_CAMERA_PX])
    ok = traj.valid
    res = traj.centers[ok] - gps_centers[ok]
    gps_rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))) if ok.any() else float("inf")
    nrms = float(np.sqrt(np.mean((res / gps_sigma[ok]) ** 2))) if ok.any() else float("inf")
    return TrajectoryMetrics(float(np.median(e)), float(np.percentile(e, 90)), gps_rms, nrms,
                             float(ok.mean()), used)


def _failures(m: TrajectoryMetrics, world_ok: bool, cfg: VerifierConfig) -> list[str]:
    out = []
    if not world_ok:
        out.append("it is not placed in metric ENU coordinates")
    if m.valid_fraction < cfg.min_valid_fraction:
        out.append(f"only {m.valid_fraction:.0%} of its poses are real (the rest are placeholders)")
    if m.reproj_p90_px > cfg.max_reproj_p90_px:
        out.append(f"90% reprojection error is {m.reproj_p90_px:.1f} px (limit {cfg.max_reproj_p90_px:g} px)")
    if m.gps_nrms > cfg.max_gps_nrms:
        out.append(f"camera positions sit {m.gps_nrms:.1f}σ from GPS on average "
                   f"(limit {cfg.max_gps_nrms:g}σ, {m.gps_rms_m:.1f} m RMS)")
    return out


def verify(ai: Trajectory, classical: ReconstructionResult, keypoints: Sequence[np.ndarray],
           gps_centers: np.ndarray, gps_sigma: np.ndarray,
           cfg: VerifierConfig | None = None) -> VerificationResult:
    cfg = cfg or VerifierConfig()
    if len(ai) != len(classical.trajectory):
        raise DataError("verifier: AI and classical trajectories cover different keyframes")
    tracks = classical.sparse.tracks
    m_ai = evaluate_trajectory(ai, keypoints, tracks, gps_centers, gps_sigma, cfg)
    m_cl = evaluate_trajectory(classical.trajectory, keypoints, tracks, gps_centers, gps_sigma, cfg)

    fails = _failures(m_ai, ai.world == "enu" and ai.metric, cfg)
    label = "stand-in AI track" if ai.is_mock else "AI track"
    if not fails:
        reasons = (f"Using the {label} ({ai.source}): reprojection {m_ai.reproj_p90_px:.1f} px at 90%, "
                   f"GPS agreement {m_ai.gps_nrms:.1f}σ.",)
        return VerificationResult("ai", ai, m_ai, m_cl, reasons)
    reasons = (f"Using the classical track ({classical.trajectory.source}) because the {label} failed: "
               + "; ".join(fails) + ".",)
    cl_fails = _failures(m_cl, True, cfg)
    if cl_fails:
        reasons += ("The classical track is also outside limits (" + "; ".join(cl_fails)
                    + "); expect reduced accuracy.",)
    return VerificationResult("classical", classical.trajectory, m_ai, m_cl, reasons)
