"""Camera self-calibration and rolling-shutter assessment.

Self-calibration: one shared focal length and radial distortion (k1, k2) plus a
relative pose per image pair are fitted by minimising the Sampson epipolar
error of verified matches, with weak priors. The principal point is held at the
image centre (it is poorly observable from a single pass). The Jacobian gives
the focal-length uncertainty; if the footage cannot pin it down (flat terrain
with little rotation is the usual cause) the prior is kept and we say so.

Rolling shutter: the readout time is taken from the camera spec (config); it is
not estimated from images. We measure how far the image moves during readout
and report whether rolling shutter needs modelling in pose estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from aerometric.config import CalibrationConfig
from aerometric.sync.offset import FlowSignal
from aerometric.types import CameraIntrinsics, DataError


@dataclass(frozen=True)
class PairCorrespondences:
    pts0: np.ndarray  # Nx2 pixels (distorted, as observed)
    pts1: np.ndarray  # Nx2


@dataclass(frozen=True)
class CalibrationResult:
    intrinsics: CameraIntrinsics
    focal_rel_std: float  # 1-sigma relative uncertainty of the focal length
    rms_px: float  # Sampson RMS at the solution (inliers of the robust fit)
    pairs_used: int
    observable: bool
    warnings: tuple[str, ...] = ()


# --------------------------------------------------------------------------- geometry helpers

def undistort_normalized(xd: np.ndarray, k1: float, k2: float, iters: int = 8) -> np.ndarray:
    """Invert x_d = x_u (1 + k1 r^2 + k2 r^4) by fixed-point iteration."""
    xu = xd.copy()
    for _ in range(iters):
        r2 = (xu ** 2).sum(1, keepdims=True)
        xu = xd / (1 + k1 * r2 + k2 * r2 ** 2)
    return xu


def distort_normalized(xu: np.ndarray, k1: float, k2: float) -> np.ndarray:
    r2 = (xu ** 2).sum(1, keepdims=True)
    return xu * (1 + k1 * r2 + k2 * r2 ** 2)


def _skew(t: np.ndarray) -> np.ndarray:
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def _direction(angles: np.ndarray) -> np.ndarray:
    th, ph = angles
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def _angles(t: np.ndarray) -> np.ndarray:
    t = t / np.linalg.norm(t)
    return np.array([np.arccos(np.clip(t[2], -1, 1)), np.arctan2(t[1], t[0])])


def _sampson(E: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    h0 = np.column_stack([x0, np.ones(len(x0))])
    h1 = np.column_stack([x1, np.ones(len(x1))])
    Ex0 = h0 @ E.T
    Etx1 = h1 @ E
    num = (h1 * Ex0).sum(1)
    den = Ex0[:, 0] ** 2 + Ex0[:, 1] ** 2 + Etx1[:, 0] ** 2 + Etx1[:, 1] ** 2
    return num / np.sqrt(den + 1e-18)


# --------------------------------------------------------------------------- self-calibration

def _init_pose(p0: np.ndarray, p1: np.ndarray, f: float, c: np.ndarray) -> np.ndarray:
    x0, x1 = (p0 - c) / f, (p1 - c) / f
    E, inl = cv2.findEssentialMat(x0, x1, np.eye(3), cv2.RANSAC, 0.999, 1.0 / f)
    if E is None:
        return np.array([0, 0, 0, np.pi / 2, 0.0])
    _, R, t, _ = cv2.recoverPose(E[:3], x0, x1, np.eye(3), mask=inl)
    return np.concatenate([Rotation.from_matrix(R).as_rotvec(), _angles(t.ravel())])


def self_calibrate(pairs: Sequence[PairCorrespondences], prior: CameraIntrinsics,
                   cfg: CalibrationConfig | None = None, seed: int = 0) -> CalibrationResult:
    cfg = cfg or CalibrationConfig()
    if prior.source == "provided" and not cfg.refine_provided:
        return CalibrationResult(prior, 0.0, float("nan"), 0, True,
                                 ("using supplied intrinsics without refinement",))

    rng = np.random.default_rng(seed)
    usable = sorted((p for p in pairs if len(p.pts0) >= 15), key=lambda p: -len(p.pts0))[: cfg.max_pairs]
    if len(usable) < 2:
        raise DataError("self-calibration: need at least 2 image pairs with 15+ verified matches")
    sub = []
    for p in usable:
        n = len(p.pts0)
        idx = rng.choice(n, min(n, cfg.max_matches_per_pair), replace=False)
        sub.append((p.pts0[idx].astype(float), p.pts1[idx].astype(float)))

    c = np.array([prior.cx, prior.cy])
    f0 = prior.fx
    k_prior = prior.dist[:2] if len(prior.dist) >= 2 else (0.0, 0.0)
    f_sigma = cfg.prior_focal_rel_sigma if prior.source != "provided" else 0.02
    poses0 = np.concatenate([_init_pose(a, b, f0, c) for a, b in sub])
    x_init = np.concatenate([[0.0, *k_prior], poses0])

    def residuals(x: np.ndarray) -> np.ndarray:
        f = f0 * np.exp(x[0])
        k1, k2 = x[1], x[2]
        out = []
        for i, (a, b) in enumerate(sub):
            pose = x[3 + 5 * i: 8 + 5 * i]
            E = _skew(_direction(pose[3:])) @ Rotation.from_rotvec(pose[:3]).as_matrix()
            xa = undistort_normalized((a - c) / f, k1, k2)
            xb = undistort_normalized((b - c) / f, k1, k2)
            out.append(_sampson(E, xa, xb) * f)
        out.append(np.array([x[0] / f_sigma, (x[1] - k_prior[0]) / cfg.prior_dist_sigma,
                             (x[2] - k_prior[1]) / cfg.prior_dist_sigma]))
        return np.concatenate(out)

    # Each pair's residuals depend only on the 3 shared intrinsics and its own 5 pose params.
    sparsity = lil_matrix((sum(len(a) for a, _ in sub) + 3, len(x_init)), dtype=int)
    row = 0
    for i, (a, _) in enumerate(sub):
        sparsity[row:row + len(a), :3] = 1
        sparsity[row:row + len(a), 3 + 5 * i: 8 + 5 * i] = 1
        row += len(a)
    sparsity[row:, :3] = 1

    sol = least_squares(residuals, x_init, loss="soft_l1", f_scale=cfg.robust_scale_px,
                        x_scale="jac", max_nfev=200, jac_sparsity=sparsity)

    res = sol.fun[:-3]
    inl = np.abs(res) < 3 * cfg.robust_scale_px
    rms = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else float("inf")
    J = sol.jac.toarray() if hasattr(sol.jac, "toarray") else sol.jac
    dof = max(1, len(sol.fun) - len(sol.x))
    s2 = max(float((sol.fun ** 2).sum()) / dof, 1e-12)
    try:
        cov = np.linalg.pinv(J.T @ J) * s2
        f_std = float(np.sqrt(max(cov[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        f_std = float("inf")

    f = f0 * float(np.exp(sol.x[0]))
    warnings: list[str] = []
    observable = f_std <= cfg.max_focal_rel_std and sol.success
    if not observable:
        warnings.append(f"the focal length could not be pinned down from this footage "
                        f"(±{f_std:.0%}); keeping the prior. Flat terrain with little camera "
                        "rotation is the usual cause; supplying intrinsics will help.")
        return CalibrationResult(prior, f_std, rms, len(sub), False, tuple(warnings))

    k1, k2 = float(sol.x[1]), float(sol.x[2])
    est = CameraIntrinsics(prior.width, prior.height, f, f, prior.cx, prior.cy,
                           model="opencv", dist=(k1, k2, 0.0, 0.0), source="estimated")
    if abs(f / f0 - 1) > 2 * f_sigma:
        warnings.append(f"estimated focal length differs from the prior by {f / f0 - 1:+.0%}")
    return CalibrationResult(est, f_std, rms, len(sub), True, tuple(warnings))


# --------------------------------------------------------------------------- rolling shutter

@dataclass(frozen=True)
class RollingShutterAssessment:
    readout_s: float  # 0 when unknown
    median_skew_px: float  # image motion during one readout
    p95_skew_px: float
    applicable: bool  # True -> pose estimation should model rolling shutter
    message: str


def assess_rolling_shutter(flow: FlowSignal, cfg: CalibrationConfig | None = None
                           ) -> RollingShutterAssessment:
    cfg = cfg or CalibrationConfig()
    speed = flow.translation_px_s[np.isfinite(flow.translation_px_s)]
    if speed.size == 0:
        return RollingShutterAssessment(cfg.readout_s, float("nan"), float("nan"), False,
                                        "No image motion measured; rolling shutter not assessed.")
    readout = cfg.readout_s if cfg.readout_s > 0 else cfg.typical_readout_s
    med, p95 = (float(np.percentile(speed, q)) * readout for q in (50, 95))
    matters = p95 > cfg.rs_threshold_px
    if cfg.readout_s <= 0:
        msg = (f"Readout time unknown. With a typical {readout * 1000:.0f} ms readout the image "
               f"would move up to {p95:.1f} px while a frame is read.")
        msg += (" Set calibration.readout_s from the camera spec so it can be modelled."
                if matters else " That is small enough to ignore.")
        return RollingShutterAssessment(0.0, med, p95, False, msg)
    msg = (f"Image moves up to {p95:.1f} px during the {readout * 1000:.0f} ms readout; "
           + ("rolling shutter will be modelled." if matters else "treating the camera as global shutter."))
    return RollingShutterAssessment(readout, med, p95, matters, msg)
