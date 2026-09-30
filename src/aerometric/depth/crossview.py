"""Cross-view consistency of aligned depth maps.

Every pixel of a keyframe's depth map is lifted to 3-D and checked against
neighbouring keyframes' depth maps (forward-backward geometric check):
  1. project the 3-D point into the neighbour and read the neighbour's depth there;
  2. the two depths must agree within `depth_tol` (relative);
  3. lifting the neighbour's depth back and projecting into this frame must land
     within `reproj_tol_px` of the starting pixel.
A neighbour only counts as evidence where the two viewing rays differ by at
least `min_angle_deg`; nearly identical viewpoints would agree about anything.

The per-pixel agreement score (consistent neighbours / informative neighbours)
is evidence for the trust engine, not a label. Errors shared by all views, as
with a stand-in depth model built from the same sparse points, are not caught.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import map_coordinates

from aerometric.config import DepthConfig
from aerometric.depth.mono import AlignedDepth
from aerometric.pose.types import CameraPose, Trajectory
from aerometric.types import DataError


@dataclass(frozen=True)
class CrossViewScore:
    agreement: np.ndarray  # HxW in [0, 1], NaN where no informative neighbour
    informative: np.ndarray  # HxW int, neighbours that could check the pixel
    consistent: np.ndarray  # HxW int
    neighbours: tuple[int, ...]  # keyframe positions used

    @property
    def checked_fraction(self) -> float:
        return float(np.mean(self.informative > 0))

    def agreeing_fraction(self, threshold: float = 0.5) -> float:
        a = self.agreement[np.isfinite(self.agreement)]
        return float(np.mean(a >= threshold)) if a.size else 0.0


def _work_intrinsics(traj: Trajectory, depth: AlignedDepth) -> tuple[float, float, float, float]:
    s = 1.0 / depth.scale_to_full
    K = traj.intrinsics
    return K.fx * s, K.fy * s, K.cx * s, K.cy * s


def _lift(depth: np.ndarray, pose: CameraPose, k: tuple[float, float, float, float],
          u: np.ndarray, v: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = k
    Xc = np.column_stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth])
    return (Xc - pose.t_cw) @ pose.R_cw  # world = R^T (Xc - t)


def _project(X: np.ndarray, pose: CameraPose, k: tuple[float, float, float, float]):
    fx, fy, cx, cy = k
    Xc = X @ pose.R_cw.T + pose.t_cw
    z = Xc[:, 2]
    zs = np.where(z > 1e-6, z, np.nan)
    return fx * Xc[:, 0] / zs + cx, fy * Xc[:, 1] / zs + cy, z


def score_frame(traj: Trajectory, depths: Sequence[AlignedDepth | None], ref: int,
                cfg: DepthConfig | None = None) -> CrossViewScore:
    cfg = cfg or DepthConfig()
    D = depths[ref]
    if D is None or not D.good:
        raise DataError(f"cross-view: keyframe {ref} has no usable depth map")
    h, w = D.depth.shape
    k_ref = _work_intrinsics(traj, D)
    v, u = np.mgrid[0:h, 0:w].astype(float)
    u, v = u.ravel(), v.ravel()
    d_ref = D.depth.ravel().astype(float)
    X = _lift(d_ref, traj.poses[ref], k_ref, u, v)
    C_ref = traj.poses[ref].center

    lo, hi = max(0, ref - cfg.neighbours), min(len(depths), ref + cfg.neighbours + 1)
    nbrs = tuple(n for n in range(lo, hi) if n != ref and depths[n] is not None and depths[n].good
                 and traj.valid[n] and depths[n].depth.shape == D.depth.shape)
    informative = np.zeros(h * w, int)
    consistent = np.zeros(h * w, int)
    cos_min = np.cos(np.radians(cfg.min_angle_deg))
    for n in nbrs:
        Dn, pose_n = depths[n], traj.poses[n]
        k_n = _work_intrinsics(traj, Dn)
        un, vn, z = _project(X, pose_n, k_n)
        inside = np.isfinite(un) & (un >= 0) & (un <= w - 1) & (vn >= 0) & (vn <= h - 1) & np.isfinite(d_ref)
        dn = np.full(h * w, np.nan)
        if inside.any():
            dn[inside] = map_coordinates(Dn.depth.astype(float), [vn[inside], un[inside]], order=1,
                                         mode="nearest")  # bilinear; NaN holes stay NaN
        a = X - C_ref
        b = X - pose_n.center
        cosang = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        info = inside & np.isfinite(dn) & (cosang <= cos_min)
        if not info.any():
            continue
        Xb = _lift(dn[info], pose_n, k_n, un[info], vn[info])
        ub, vb, _ = _project(Xb, traj.poses[ref], k_ref)
        reproj = np.hypot(ub - u[info], vb - v[info])
        rel = np.abs(dn[info] - z[info]) / z[info]
        ok = (rel < cfg.depth_tol) & (reproj < cfg.reproj_tol_px)
        informative[info] += 1
        idx = np.flatnonzero(info)
        consistent[idx[ok]] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        agreement = np.where(informative > 0, consistent / informative, np.nan)
    return CrossViewScore(agreement.reshape(h, w), informative.reshape(h, w), consistent.reshape(h, w), nbrs)


def score_all(traj: Trajectory, depths: Sequence[AlignedDepth | None],
              cfg: DepthConfig | None = None) -> list[CrossViewScore | None]:
    return [score_frame(traj, depths, i, cfg) if d is not None and d.good and traj.valid[i] else None
            for i, d in enumerate(depths)]
