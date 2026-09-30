"""TSDF fusion of aligned depth maps into the metric mesh (built in; Open3D not required).

Each keyframe's metric depth map is integrated into a truncated signed distance
volume over the scene. Pixels are weighted by their cross-view agreement
(unchecked pixels count less, disagreeing ones are skipped) and pixels the
segmenter marks as moving objects, sky or water are left out. The zero level
set is extracted with marching cubes. Mesh vertices then take semantic labels
by the same visibility-checked voting used for sparse points.

Poisson reconstruction needs Open3D, which is not installed; asking for it
raises a clear error rather than silently falling back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes

from dronerecon.config import FusionConfig, SemanticConfig
from dronerecon.depth.crossview import CrossViewScore
from dronerecon.depth.mono import AlignedDepth
from dronerecon.fusion.semantic import SemanticPoints, vote_labels
from dronerecon.pose.types import Trajectory
from dronerecon.segmentation import SegmentationResult
from dronerecon.types import DataError


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray  # Vx3 ENU metres
    faces: np.ndarray  # Fx3 int
    normals: np.ndarray  # Vx3
    vertex_weight: np.ndarray  # V, fused observation weight around each vertex
    semantics: SemanticPoints | None
    voxel_m: float
    backend: str
    warnings: tuple[str, ...] = ()


@dataclass
class _Volume:
    origin: np.ndarray  # world position of voxel (0,0,0) centre
    voxel: float
    shape: tuple[int, int, int]
    tsdf: np.ndarray
    weight: np.ndarray


def _frame_pixels(k: int, traj: Trajectory, depths, scores, segs, cfg: FusionConfig):
    """Depth map with excluded pixels set to NaN, and per-pixel weights."""
    d = depths[k]
    depth = d.depth.astype(float).copy()
    weight = np.ones_like(depth)
    sc = scores[k] if scores is not None else None
    if sc is not None:
        a = sc.agreement
        weight = np.where(np.isfinite(a), a, cfg.unscored_weight)
        depth[np.isfinite(a) & (a < cfg.min_agreement)] = np.nan
    else:
        weight[:] = cfg.unscored_weight
    if segs is not None and segs[k] is not None:
        ex = segs[k].exclusion_mask(dilate_px=3)
        ex = cv2.resize(ex.astype(np.uint8), depth.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        depth[ex] = np.nan
    return depth, weight


def _bounds(traj: Trajectory, depths, frames) -> tuple[np.ndarray, np.ndarray, float]:
    pts, gsd = [], []
    K = traj.intrinsics
    for k in frames:
        d = depths[k]
        s = 1.0 / d.scale_to_full
        h, w = d.depth.shape
        v, u = np.mgrid[0:h:4, 0:w:4]
        z = d.depth[v, u].astype(float)
        ok = np.isfinite(z)
        if not ok.any():
            continue
        u, v, z = u[ok], v[ok], z[ok]
        Xc = np.column_stack([(u - K.cx * s) / (K.fx * s) * z, (v - K.cy * s) / (K.fy * s) * z, z])
        p = traj.poses[k]
        pts.append((Xc - p.t_cw) @ p.R_cw)
        gsd.append(np.median(z) / (K.fx * s))
    if not pts:
        raise DataError("fusion: no valid depth to fuse")
    P = np.concatenate(pts)
    lo, hi = np.percentile(P, 0.5, axis=0), np.percentile(P, 99.5, axis=0)
    return lo, hi, float(np.median(gsd))


def _integrate(vol: _Volume, traj: Trajectory, k: int, depth: np.ndarray, pix_w: np.ndarray,
               scale_to_full: float, trunc: float, chunk: int) -> None:
    K = traj.intrinsics
    s = 1.0 / scale_to_full
    fx, fy, cx, cy = K.fx * s, K.fy * s, K.cx * s, K.cy * s
    p = traj.poses[k]
    h, w = depth.shape
    nx, ny, nz = vol.shape
    flat_t, flat_w = vol.tsdf.reshape(-1), vol.weight.reshape(-1)
    total = nx * ny * nz
    for start in range(0, total, chunk):
        idx = np.arange(start, min(total, start + chunk))
        ijk = np.column_stack(np.unravel_index(idx, vol.shape))
        X = vol.origin + ijk * vol.voxel
        Xc = X @ p.R_cw.T + p.t_cw
        z = Xc[:, 2]
        front = z > 1e-3
        if not front.any():
            continue
        u = fx * Xc[:, 0] / np.where(front, z, 1) + cx
        v = fy * Xc[:, 1] / np.where(front, z, 1) + cy
        inside = front & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        if not inside.any():
            continue
        ui, vi = np.round(u[inside]).astype(int), np.round(v[inside]).astype(int)
        dm = depth[vi, ui]
        sdf = dm - z[inside]
        ok = np.isfinite(dm) & (sdf > -trunc)
        if not ok.any():
            continue
        tgt = idx[inside][ok]
        val = np.clip(sdf[ok] / trunc, -1, 1)
        wt = pix_w[vi[ok], ui[ok]]
        W0 = flat_w[tgt]
        flat_t[tgt] = (flat_t[tgt] * W0 + val * wt) / np.maximum(W0 + wt, 1e-9)
        flat_w[tgt] = W0 + wt


def _drop_unobserved_crossings(verts, faces, normals, observed):
    """Remove triangles whose zero crossing touches a never-observed voxel.

    Voxels behind a surface keep their initial +1 and sit next to observed -1
    voxels; marching cubes would otherwise put a phantom surface there.
    """
    lo = np.floor(verts).astype(int)
    hi = np.minimum(np.ceil(verts).astype(int), np.array(observed.shape) - 1)
    good_v = observed[lo[:, 0], lo[:, 1], lo[:, 2]] & observed[hi[:, 0], hi[:, 1], hi[:, 2]]
    keep_f = good_v[faces].all(axis=1)
    faces = faces[keep_f]
    used = np.unique(faces)
    remap = np.full(len(verts), -1)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces], normals[used]


def fuse_tsdf(traj: Trajectory, depths: Sequence[AlignedDepth | None],
              scores: Sequence[CrossViewScore | None] | None = None,
              segs: Sequence[SegmentationResult | None] | None = None,
              cfg: FusionConfig | None = None, semantic_cfg: SemanticConfig | None = None) -> Mesh:
    cfg = cfg or FusionConfig()
    if cfg.method == "poisson":
        raise DataError("Poisson reconstruction needs Open3D, which is not installed; use fusion.method 'tsdf'")
    if cfg.method != "tsdf":
        raise DataError(f"fusion.method must be 'tsdf' or 'poisson', got '{cfg.method}'")
    if traj.world != "enu" or not traj.metric:
        raise DataError("fusion: trajectory must be metric ENU (run georeferencing first)")
    frames = [k for k, d in enumerate(depths) if d is not None and d.good and traj.valid[k]]
    if not frames:
        raise DataError("fusion: no keyframe has a usable depth map")

    lo, hi, gsd = _bounds(traj, depths, frames)
    voxel = cfg.voxel_m if cfg.voxel_m > 0 else max(2 * gsd, 0.05)
    warnings: list[str] = []
    while np.prod(np.ceil((hi - lo) / voxel) + 1) > cfg.max_voxels:
        voxel *= 1.25
    if cfg.voxel_m > 0 and voxel > cfg.voxel_m:
        warnings.append(f"voxel size raised from {cfg.voxel_m:g} m to {voxel:.2f} m to fit the memory limit")
    trunc = cfg.trunc_voxels * voxel
    lo, hi = lo - 2 * trunc, hi + 2 * trunc
    shape = tuple(int(v) for v in np.ceil((hi - lo) / voxel) + 1)
    vol = _Volume(lo, voxel, shape, np.ones(shape, np.float32), np.zeros(shape, np.float32))

    for k in frames:
        depth, pix_w = _frame_pixels(k, traj, depths, scores, segs, cfg)
        _integrate(vol, traj, k, depth, pix_w, depths[k].scale_to_full, trunc, cfg.chunk_voxels)

    observed = vol.weight > 0
    if not observed.any() or vol.tsdf[observed].min() >= 0 or vol.tsdf[observed].max() <= 0:
        raise DataError("fusion: the depth maps do not define a surface inside the volume")
    verts, faces, normals, _ = marching_cubes(vol.tsdf, level=0.0, mask=observed, allow_degenerate=False)
    verts, faces, normals = _drop_unobserved_crossings(verts, faces, normals, observed)
    if len(faces) == 0:
        raise DataError("fusion: no observed surface could be extracted")
    world = lo + verts * voxel
    vw = map_coordinates(vol.weight, verts.T, order=1, mode="nearest")
    # marching_cubes normals point toward increasing tsdf, i.e. out of the surface toward free space
    sem = vote_labels(world, traj, segs, depths, semantic_cfg) if segs is not None else None
    if scores is None:
        warnings.append("no cross-view scores were given; every depth pixel was fused with reduced weight")
    return Mesh(world, faces.astype(np.int64), normals, vw, sem, voxel, "builtin-tsdf", tuple(warnings))
