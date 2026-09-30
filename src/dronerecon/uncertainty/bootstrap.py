"""Bootstrap re-runs: how much does each point move when the evidence changes?

Each run keeps a random subset of keyframes and perturbs the GPS fixes by their
own uncertainty, then re-runs the reconstruction. The spread of a point across
runs is its (uncalibrated) uncertainty; calibration against held-out views
comes next.

Two re-run steps are provided and chained:
  * pose run: bundle adjustment on the subset with perturbed GPS, then the
    ORIGINAL tracks are re-triangulated, so every sparse point keeps its identity;
  * surface run: stored depth predictions are re-aligned to that run's sparse
    points and re-fused; each original vertex's signed offset along its normal
    to the run's surface is recorded (missing if no surface within max_gap_m).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from dronerecon.config import ClassicalConfig, DepthConfig, FusionConfig, UncertaintyConfig
from dronerecon.depth.mono import DepthPrediction, align_depth, frame_observations
from dronerecon.fusion.tsdf import fuse_tsdf
from dronerecon.pose.classical import reconstruct
from dronerecon.pose.geometry import normalized_points
from dronerecon.pose.tracks import PairMatches, triangulate
from dronerecon.pose.types import CameraPose, Trajectory
from dronerecon.types import DataError


@dataclass(frozen=True)
class RunDraw:
    keep: np.ndarray  # bool per keyframe
    gps: np.ndarray  # Nx3 perturbed GPS centres


@dataclass(frozen=True)
class BootstrapStats:
    mean_offset: np.ndarray  # N (surface: signed metres along normal) or Nx3 (points)
    std: np.ndarray  # N, 1-sigma metres (NaN where too few runs)
    valid_runs: np.ndarray  # N int
    n_runs: int


def draw_runs(n_keyframes: int, gps: np.ndarray, gps_sigma: np.ndarray, cfg: UncertaintyConfig,
              seed: int = 0) -> list[RunDraw]:
    if n_keyframes < 3:
        raise DataError("bootstrap: need at least 3 keyframes")
    rng = np.random.default_rng(seed)
    k = max(3, int(round(cfg.keep_fraction * n_keyframes)))
    draws = []
    for _ in range(cfg.n_runs):
        keep = np.zeros(n_keyframes, bool)
        keep[rng.choice(n_keyframes, k, replace=False)] = True
        noisy = gps + rng.normal(0.0, 1.0, gps.shape) * gps_sigma * cfg.gps_perturb_scale
        draws.append(RunDraw(keep, noisy))
    return draws


def summarise(samples: np.ndarray, cfg: UncertaintyConfig) -> BootstrapStats:
    """samples: R x N (scalar offsets) or R x N x 3 (positions); NaN = missing in that run."""
    valid = np.all(np.isfinite(samples), axis=-1) if samples.ndim == 3 else np.isfinite(samples)
    n_valid = valid.sum(0)
    with warnings.catch_warnings():  # points never (or once) measured are expected; they get NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(samples, axis=0)
        if samples.ndim == 3:
            var = np.nanvar(samples, axis=0, ddof=1).sum(-1) / 3.0  # mean per-axis variance
        else:
            var = np.nanvar(samples, axis=0, ddof=1)
    std = np.where(n_valid >= cfg.min_valid_runs, np.sqrt(var), np.nan)
    return BootstrapStats(mean, std, n_valid, samples.shape[0])


# --------------------------------------------------------------------------- pose run

def subset_trajectory(traj: Trajectory, keep: np.ndarray) -> Trajectory:
    idx = np.flatnonzero(keep)
    return replace(traj, frame_indices=traj.frame_indices[idx], t_video_s=traj.t_video_s[idx],
                   poses=tuple(traj.poses[i] for i in idx), valid=traj.valid[idx])


def retriangulate(tracks: Sequence[np.ndarray], keypoints: Sequence[np.ndarray],
                  poses: Sequence[CameraPose | None], traj: Trajectory) -> np.ndarray:
    """Original tracks triangulated with (possibly partial) poses; NaN where < 2 posed views."""
    K = traj.intrinsics
    norm = [normalized_points(k.astype(float), K) if len(k) else np.empty((0, 2)) for k in keypoints]
    out = np.full((len(tracks), 3), np.nan)
    for i, tr in enumerate(tracks):
        tr = tr[[poses[c] is not None for c in tr[:, 0]]]
        if len(tr) < 2:
            continue
        obs = np.array([norm[c][k] for c, k in tr])
        out[i] = triangulate(poses, obs, tr[:, 0])
    return out


@dataclass(frozen=True)
class PoseRun:
    poses: list[CameraPose | None]  # full-length, None for dropped keyframes
    points: np.ndarray  # original tracks re-triangulated (NaN where not possible)


def pose_run(draw: RunDraw, keypoints: Sequence[np.ndarray], pairs: Sequence[PairMatches], initial: Trajectory,
             gps_sigma: np.ndarray, tracks: Sequence[np.ndarray], cfg: ClassicalConfig | None = None) -> PoseRun:
    idx = np.flatnonzero(draw.keep)
    remap = {int(o): n for n, o in enumerate(idx)}
    sub_pairs = [PairMatches(remap[p.i], remap[p.j], p.idx_i, p.idx_j) for p in pairs
                 if p.i in remap and p.j in remap]
    res = reconstruct([keypoints[i] for i in idx], sub_pairs, subset_trajectory(initial, draw.keep),
                      draw.gps[idx], gps_sigma[idx], cfg)
    poses: list[CameraPose | None] = [None] * len(initial)
    for n, o in enumerate(idx):
        poses[o] = res.trajectory.poses[n]
    return PoseRun(poses, retriangulate(tracks, keypoints, poses, initial))


# --------------------------------------------------------------------------- surface run

def surface_offsets(query: np.ndarray, normals: np.ndarray, run_vertices: np.ndarray, max_gap: float) -> np.ndarray:
    """Signed distance along each query normal to the nearest run vertex (NaN beyond max_gap)."""
    if len(run_vertices) == 0:
        return np.full(len(query), np.nan)
    dist, j = cKDTree(run_vertices).query(query, distance_upper_bound=max_gap)
    ok = np.isfinite(dist)
    out = np.full(len(query), np.nan)
    n = normals[ok] / (np.linalg.norm(normals[ok], axis=1, keepdims=True) + 1e-12)
    out[ok] = np.einsum("ij,ij->i", run_vertices[j[ok]] - query[ok], n)
    return out


def surface_run(run: PoseRun, base: Trajectory, predictions: Sequence[DepthPrediction | None],
                tracks: Sequence[np.ndarray], keypoints: Sequence[np.ndarray], full_width: int,
                depth_cfg: DepthConfig | None = None, fusion_cfg: FusionConfig | None = None):
    """Re-align stored depth predictions to this run's geometry and re-fuse. Returns mesh vertices."""
    kept = [i for i, p in enumerate(run.poses) if p is not None]
    traj = replace(base, poses=tuple(run.poses[i] if run.poses[i] is not None else base.poses[i]
                                     for i in range(len(base))),
                   valid=np.array([p is not None for p in run.poses]) & base.valid)
    ok_pts = np.all(np.isfinite(run.points), axis=1)
    pts = np.where(ok_pts[:, None], run.points, 0.0)
    live_tracks = [t if ok else t[:0] for t, ok in zip(tracks, ok_pts)]
    depths = [None] * len(base)
    for i in kept:
        if predictions[i] is None:
            continue
        obs = frame_observations(traj, pts, live_tracks, keypoints, i)
        depths[i] = align_depth(predictions[i], obs, full_width, depth_cfg)
    mesh = fuse_tsdf(traj, depths, None, None, fusion_cfg)
    return mesh.vertices


# --------------------------------------------------------------------------- driver

def run_bootstrap(draws: Sequence[RunDraw], run_one: Callable[[RunDraw], np.ndarray],
                  cfg: UncertaintyConfig) -> BootstrapStats:
    """run_one(draw) -> per-point samples (N or Nx3, NaN where missing). Failed runs are skipped."""
    samples = []
    for d in draws:
        try:
            samples.append(run_one(d))
        except DataError:
            continue  # e.g. a subset too sparse to reconstruct; it simply contributes nothing
    if not samples:
        raise DataError("bootstrap: every run failed")
    return summarise(np.stack(samples), cfg)
