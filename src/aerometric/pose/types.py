"""Pose data shared by the AI track, the classical track and the verifier.

Camera convention (OpenCV): x right, y down, z forward. `R_cw`, `t_cw` map
world points into the camera: X_c = R_cw @ X_w + t_cw.
World is either a local frame (arbitrary gauge) or ENU metres from the
smoothed-GPS origin, as stated by `Trajectory.world`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from aerometric.types import CameraIntrinsics, DataError

World = Literal["local", "enu"]


@dataclass(frozen=True)
class CameraPose:
    R_cw: np.ndarray  # 3x3
    t_cw: np.ndarray  # 3

    @property
    def center(self) -> np.ndarray:
        return -self.R_cw.T @ self.t_cw

    @classmethod
    def from_center(cls, R_cw: np.ndarray, center: np.ndarray) -> "CameraPose":
        return cls(R_cw, -R_cw @ center)


@dataclass(frozen=True)
class Similarity:
    """X' = s R X + t"""

    s: float
    R: np.ndarray
    t: np.ndarray

    def apply_points(self, X: np.ndarray) -> np.ndarray:
        return self.s * X @ self.R.T + self.t

    def apply_pose(self, p: CameraPose) -> CameraPose:
        R_cw = p.R_cw @ self.R.T
        return CameraPose.from_center(R_cw, self.apply_points(p.center[None])[0])


@dataclass(frozen=True)
class Trajectory:
    frame_indices: np.ndarray  # N video frame indices (keyframes)
    t_video_s: np.ndarray  # N
    poses: tuple[CameraPose, ...]
    valid: np.ndarray  # N bool; False = pose is a placeholder carried over a broken link
    intrinsics: CameraIntrinsics
    world: World
    metric: bool  # True when scale is in metres
    source: str  # e.g. "mock-feedforward", "pycolmap"
    is_mock: bool
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.frame_indices)
        if not (len(self.t_video_s) == len(self.poses) == len(self.valid) == n):
            raise DataError("Trajectory: frame_indices, t_video_s, poses and valid must match in length")

    def __len__(self) -> int:
        return len(self.poses)

    @property
    def centers(self) -> np.ndarray:
        return np.array([p.center for p in self.poses])

    def transformed(self, sim: Similarity, world: World, metric: bool) -> "Trajectory":
        return Trajectory(self.frame_indices, self.t_video_s, tuple(sim.apply_pose(p) for p in self.poses),
                          self.valid, self.intrinsics, world, metric, self.source, self.is_mock, self.warnings)
