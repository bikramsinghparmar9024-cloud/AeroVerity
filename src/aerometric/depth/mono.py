"""Monocular depth prediction and its alignment to the sparse model.

Monocular networks predict depth (or inverse depth) only up to an unknown scale
and shift. Each keyframe's prediction is aligned to the metric sparse points
that frame observes with a robust fit:
    depth kind:          d = a * p + b
    inverse_depth kind:  1/d = a * p + b
giving a metric depth map plus how well it agreed with the sparse points.

Backends: Depth Anything V2 (depth_anything.py, used when its weights are on
disk) and a labelled STAND-IN that interpolates the frame's own sparse depths
across the image. The stand-in is smooth between points and knows nothing the
sparse model does not; the cross-view check that follows catches where either
backend is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import least_squares

from aerometric.config import DepthConfig
from aerometric.pose.types import Trajectory
from aerometric.types import DataError

DepthKind = Literal["depth", "inverse_depth"]


@dataclass(frozen=True)
class FrameObservations:
    """Sparse points seen by one keyframe: pixel positions (full resolution) and metric depth."""

    uv: np.ndarray  # Nx2
    depth: np.ndarray  # N, metres along the optical axis
    point_ids: np.ndarray  # N, index into the sparse model


@dataclass(frozen=True)
class DepthPrediction:
    values: np.ndarray  # HxW float32 at work resolution
    kind: DepthKind
    backend: str
    is_mock: bool


@dataclass(frozen=True)
class AlignedDepth:
    depth: np.ndarray  # HxW metres at work resolution, NaN where invalid
    scale_to_full: float  # full-resolution pixels per work pixel
    a: float
    b: float
    kind: DepthKind
    median_rel_error: float  # on the sparse points, |d_pred - d_sparse| / d_sparse
    points_used: int
    good: bool
    backend: str
    is_mock: bool


class DepthModel(Protocol):
    name: str
    is_mock: bool

    def predict(self, image: np.ndarray, obs: FrameObservations, full_width: int) -> DepthPrediction:
        """image: BGR at work resolution; obs: full-resolution pixels. Returns a map the size of `image`."""
        ...


# --------------------------------------------------------------------------- sparse observations

def frame_observations(traj: Trajectory, points: np.ndarray, tracks, keypoints, frame: int) -> FrameObservations:
    uv, ids = [], []
    for pid, tr in enumerate(tracks):
        hit = tr[tr[:, 0] == frame]
        if len(hit):
            uv.append(keypoints[frame][hit[0, 1]])
            ids.append(pid)
    if not ids:
        return FrameObservations(np.empty((0, 2)), np.empty(0), np.empty(0, int))
    ids = np.array(ids)
    p = traj.poses[frame]
    z = (points[ids] @ p.R_cw.T + p.t_cw)[:, 2]
    keep = z > 0
    return FrameObservations(np.asarray(uv, float)[keep], z[keep], ids[keep])


# --------------------------------------------------------------------------- stand-in backend

class SparseInterpolationStandIn:
    """PLACEHOLDER for a monocular depth network: interpolates sparse depths across the frame."""

    name = "mock-sparse-interp"
    is_mock = True

    def predict(self, image: np.ndarray, obs: FrameObservations, full_width: int) -> DepthPrediction:
        h, w = image.shape[:2]
        s = w / full_width
        if len(obs.depth) < 3:
            return DepthPrediction(np.full((h, w), np.nan, np.float32), "depth", self.name, True)
        uv = obs.uv * s
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        q = np.column_stack([u.ravel(), v.ravel()])
        lin = LinearNDInterpolator(uv, obs.depth)(q)
        near = NearestNDInterpolator(uv, obs.depth)(q)
        out = np.where(np.isfinite(lin), lin, near).reshape(h, w)
        return DepthPrediction(out.astype(np.float32), "depth", self.name, True)


def make_depth_model(cfg: DepthConfig | None = None) -> DepthModel:
    """'auto': Depth Anything V2 when its weights are on disk, otherwise the stand-in."""
    cfg = cfg or DepthConfig()
    if cfg.backend == "mock":
        return SparseInterpolationStandIn()
    if cfg.backend in ("depth-anything", "auto"):
        from aerometric.depth.depth_anything import DepthAnythingV2, weights_cached

        if cfg.backend == "depth-anything" or weights_cached(cfg):
            return DepthAnythingV2(cfg)
        return SparseInterpolationStandIn()
    raise DataError(f"depth backend '{cfg.backend}' is not available; use 'auto', 'depth-anything' or 'mock'")


# --------------------------------------------------------------------------- alignment

def align_depth(pred: DepthPrediction, obs: FrameObservations, full_width: int,
                cfg: DepthConfig | None = None) -> AlignedDepth:
    cfg = cfg or DepthConfig()
    h, w = pred.values.shape
    s = w / full_width
    nan_map = np.full((h, w), np.nan, np.float32)

    def fail(n: int) -> AlignedDepth:
        return AlignedDepth(nan_map, 1 / s, float("nan"), float("nan"), pred.kind, float("inf"), n, False,
                            pred.backend, pred.is_mock)

    if len(obs.depth) < cfg.min_points:
        return fail(len(obs.depth))
    px = np.clip(np.round(obs.uv * s).astype(int), [0, 0], [w - 1, h - 1])
    p = pred.values[px[:, 1], px[:, 0]].astype(float)
    ok = np.isfinite(p)
    p, d = p[ok], obs.depth[ok]
    if len(d) < cfg.min_points:
        return fail(len(d))

    target = d if pred.kind == "depth" else 1.0 / d
    scale = cfg.huber_rel * float(np.median(np.abs(target)))
    A = np.column_stack([p, np.ones_like(p)])
    x0 = np.linalg.lstsq(A, target, rcond=None)[0]
    sol = least_squares(lambda x: A @ x - target, x0, loss="huber", f_scale=scale)
    a, b = map(float, sol.x)

    fitted = a * pred.values.astype(float) + b
    depth = fitted if pred.kind == "depth" else np.where(fitted > 0, 1.0 / np.where(fitted > 0, fitted, 1), np.nan)
    depth = np.where(depth > 0, depth, np.nan).astype(np.float32)

    at_pts = a * p + b
    at_pts = at_pts if pred.kind == "depth" else 1.0 / np.where(at_pts > 0, at_pts, np.nan)
    rel = np.abs(at_pts - d) / d
    med = float(np.nanmedian(rel)) if np.isfinite(rel).any() else float("inf")
    good = med <= cfg.max_rel_residual and a > 0
    if good and np.std(target) > cfg.min_depth_variation * abs(float(np.mean(target))):
        # the prediction must actually carry the scene's depth structure, not just its average
        good = np.std(p) > 0 and float(np.corrcoef(p, target)[0, 1]) >= cfg.min_correlation
    return AlignedDepth(depth, 1 / s, a, b, pred.kind, med, len(d), bool(good), pred.backend, pred.is_mock)


def predict_aligned(model: DepthModel, image: np.ndarray, obs: FrameObservations,
                    cfg: DepthConfig | None = None) -> AlignedDepth:
    cfg = cfg or DepthConfig()
    H, W = image.shape[:2]
    w = min(cfg.work_width, W)
    h = round(H * w / W)
    small = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA) if w < W else image
    return align_depth(model.predict(small, obs, W), obs, W, cfg)
