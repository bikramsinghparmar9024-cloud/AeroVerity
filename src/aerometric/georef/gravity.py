"""Gravity observations: the direction of world 'up' expressed in each camera's frame.

Source here: gimbal attitude from telemetry (DJI SRT gb_pitch / gb_roll, or a
flight log). Gimbal yaw is not needed for gravity. Convention assumed:
  pitch  0 deg = camera level, -90 deg = looking straight down;
  roll   positive = image rotates clockwise (right side down) * roll_sign.
The roll sign is configurable because vendors differ; confirm it on a real
flight by checking that the gravity residual drops when georeferencing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerometric.types import AttitudeSeries, DataError


@dataclass(frozen=True)
class GravityObservations:
    keyframes: np.ndarray  # positions in the trajectory (0..N-1)
    up_cam: np.ndarray  # Kx3 unit vectors: world up in camera coordinates (x right, y down, z fwd)
    sigma_deg: np.ndarray  # K


def up_from_gimbal(pitch_deg: np.ndarray, roll_deg: np.ndarray, roll_sign: float = 1.0) -> np.ndarray:
    p, r = np.radians(pitch_deg), np.radians(roll_deg) * roll_sign
    up_pitch = np.column_stack([np.zeros_like(p), -np.cos(p), np.sin(p)])  # roll = 0
    # rolling the camera clockwise by r about its optical axis rotates vectors in the
    # camera frame counter-clockwise by r
    c, s = np.cos(-r), np.sin(-r)
    x, y = up_pitch[:, 0], up_pitch[:, 1]
    return np.column_stack([c * x - s * y, s * x + c * y, up_pitch[:, 2]])


def rotation_from_gimbal(yaw_deg: float, pitch_deg: float, roll_deg: float, roll_sign: float = 1.0) -> np.ndarray:
    """Camera R_cw (world ENU -> camera) from gimbal attitude.

    yaw: compass heading the camera faces (0 north, 90 east, clockwise); pitch and roll
    as in `up_from_gimbal`, so R_cw @ (0, 0, 1) equals up_from_gimbal(pitch, roll).
    """
    h, p = np.radians(yaw_deg), np.radians(pitch_deg)
    fwd = np.array([np.sin(h), np.cos(h), 0.0])
    right = np.array([np.cos(h), -np.sin(h), 0.0])
    z = np.cos(p) * fwd + np.sin(p) * np.array([0.0, 0.0, 1.0])
    y = np.cross(z, right)
    R = np.array([right, y, z])  # rows: camera x (right), y (down), z (optical axis) in world
    r = np.radians(roll_deg) * roll_sign
    c, s = np.cos(r), np.sin(r)
    Rz = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # clockwise image roll
    return Rz @ R


def gravity_from_attitude(attitude: AttitudeSeries, t_telemetry: np.ndarray, sigma_deg: float,
                          roll_sign: float = 1.0) -> GravityObservations:
    """Gravity per keyframe from gimbal pitch/roll interpolated at telemetry times."""
    t = np.asarray(t_telemetry, float)
    inside = (t >= attitude.t[0]) & (t <= attitude.t[-1])
    if not inside.any():
        raise DataError("gravity: no keyframe falls inside the attitude log")
    pitch = np.interp(t[inside], attitude.t, attitude.ypr_deg[:, 1])
    roll = np.interp(t[inside], attitude.t, attitude.ypr_deg[:, 2])
    return GravityObservations(np.flatnonzero(inside), up_from_gimbal(pitch, roll, roll_sign),
                               np.full(int(inside.sum()), sigma_deg))
