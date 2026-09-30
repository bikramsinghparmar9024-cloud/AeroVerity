"""Fast feed-forward ("AI") pose track and the T0 preview.

A real backend (a feed-forward multi-view reconstruction network) will take
keyframe images and return camera poses in one pass. None is installed yet, so
the only backend is a labelled STAND-IN: it chains two-view essential-matrix
poses between consecutive keyframes and borrows scale from GPS baselines. It
exists so the T0 preview, the pose verifier and the viewer can be built and
tested; it is not the AI model and must not be reported as one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

import numpy as np

from aerometric.calibration import PairCorrespondences
from aerometric.config import PoseConfig
from aerometric.pose.geometry import (
    STRAIGHT_TRACK_WARNING,
    is_near_straight,
    normalized_points,
    relative_pose,
    umeyama,
)
from aerometric.pose.types import CameraPose, Trajectory
from aerometric.types import CameraIntrinsics, DataError


@dataclass(frozen=True)
class PoseInputs:
    frame_indices: np.ndarray  # keyframe video indices, flight order
    t_video_s: np.ndarray
    intrinsics: CameraIntrinsics
    consecutive: Mapping[int, PairCorrespondences]  # k -> matches between keyframe k and k+1
    gps_centers: np.ndarray | None = None  # Nx3 ENU from the smoothed track, for scale
    load_image: Callable[[int], np.ndarray] | None = None  # frame index -> BGR (real models)


class FeedForwardPoseModel(Protocol):
    name: str
    is_mock: bool

    def estimate(self, inputs: PoseInputs) -> Trajectory: ...


class ChainedTwoViewStandIn:
    """PLACEHOLDER for the feed-forward model (classical two-view chaining)."""

    name = "mock-feedforward"
    is_mock = True

    def __init__(self, cfg: PoseConfig | None = None) -> None:
        self.cfg = cfg or PoseConfig()

    def estimate(self, inputs: PoseInputs) -> Trajectory:
        n = len(inputs.frame_indices)
        if n < 2:
            raise DataError("pose: need at least 2 keyframes")
        K = inputs.intrinsics
        R, t = np.eye(3), np.zeros(3)
        poses, valid = [CameraPose(R, t)], [True]
        broken = 0
        for k in range(n - 1):
            pair = inputs.consecutive.get(k)
            rel = None
            if pair is not None and len(pair.pts0) >= self.cfg.min_pair_matches:
                rel = relative_pose(normalized_points(pair.pts0, K), normalized_points(pair.pts1, K), K.fx)
            if rel is None:
                broken += 1
                poses.append(CameraPose(R, t))  # carry the last pose; flagged invalid
                valid.append(False)
                continue
            R_rel, t_rel, _ = rel
            scale = 1.0
            if inputs.gps_centers is not None:
                scale = float(np.linalg.norm(inputs.gps_centers[k + 1] - inputs.gps_centers[k])) or 1.0
            R, t = R_rel @ R, R_rel @ t + scale * t_rel
            poses.append(CameraPose(R, t))
            valid.append(True)

        warnings = ["poses come from a stand-in (two-view chaining), not the feed-forward model"]
        if broken:
            warnings.append(f"{broken} keyframe link(s) had too few matches; those poses are placeholders")
        return Trajectory(np.asarray(inputs.frame_indices), np.asarray(inputs.t_video_s, float),
                          tuple(poses), np.array(valid), K, "local", inputs.gps_centers is not None,
                          self.name, True, tuple(warnings))


def make_pose_model(cfg: PoseConfig | None = None) -> FeedForwardPoseModel:
    cfg = cfg or PoseConfig()
    if cfg.ai_backend == "mock":
        return ChainedTwoViewStandIn(cfg)
    raise DataError(f"feed-forward pose backend '{cfg.ai_backend}' is not available; "
                    "the only backend implemented so far is 'mock'")


# --------------------------------------------------------------------------- T0 preview

def t0_preview(traj: Trajectory, gps_centers: np.ndarray | None,
               cfg: PoseConfig | None = None) -> Trajectory:
    """Place the fast trajectory in ENU by a similarity fit of camera centres to GPS.

    This is a quick placement for the preview only; the metric track and the
    georeferencing stage produce the positions used for measurement.
    """
    cfg = cfg or PoseConfig()
    if gps_centers is None:
        return traj
    ok = traj.valid & np.all(np.isfinite(gps_centers), axis=1)
    if ok.sum() < cfg.preview_min_gps_points:
        return Trajectory(traj.frame_indices, traj.t_video_s, traj.poses, traj.valid, traj.intrinsics,
                          traj.world, traj.metric, traj.source, traj.is_mock,
                          traj.warnings + ("too few GPS fixes to place the preview; shown in its own frame",))
    sim = umeyama(traj.centers[ok], gps_centers[ok])
    placed = traj.transformed(sim, "enu", True)
    if is_near_straight(gps_centers[ok], cfg.straight_track_ratio):
        placed = Trajectory(placed.frame_indices, placed.t_video_s, placed.poses, placed.valid,
                            placed.intrinsics, placed.world, placed.metric, placed.source, placed.is_mock,
                            placed.warnings + (STRAIGHT_TRACK_WARNING,))
    return placed


def preview_payload(traj: Trajectory) -> dict:
    """JSON-ready T0 preview for the viewer."""
    return {
        "kind": "t0_preview",
        "source": traj.source,
        "is_mock": traj.is_mock,
        "world": traj.world,
        "metric": traj.metric,
        "cameras": [
            {"frame": int(f), "t": float(t), "center": p.center.round(4).tolist(),
             "R_cw": p.R_cw.round(6).tolist(), "valid": bool(v)}
            for f, t, p, v in zip(traj.frame_indices, traj.t_video_s, traj.poses, traj.valid)
        ],
        "intrinsics": {"width": traj.intrinsics.width, "height": traj.intrinsics.height,
                       "fx": traj.intrinsics.fx, "fy": traj.intrinsics.fy,
                       "cx": traj.intrinsics.cx, "cy": traj.intrinsics.cy},
        "notes": list(traj.warnings),
    }
