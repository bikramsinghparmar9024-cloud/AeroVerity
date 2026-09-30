"""Small geometry helpers shared by the pose stages."""

from __future__ import annotations

import cv2
import numpy as np

from aerometric.calibration import undistort_normalized
from aerometric.pose.types import Similarity
from aerometric.types import CameraIntrinsics, DataError


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Similarity:
    """Least-squares similarity with dst ≈ s R src + t (Umeyama 1991)."""
    if src.shape != dst.shape or len(src) < 3:
        raise DataError("umeyama: need at least 3 matching points of the same shape")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = (a ** 2).sum() / len(src)
    if var < 1e-12:
        raise DataError("umeyama: source points are all at the same place")
    s = float(np.trace(np.diag(S) @ D) / var) if with_scale else 1.0
    return Similarity(s, R, mu_d - s * R @ mu_s)


def is_near_straight(centers: np.ndarray, ratio: float) -> bool:
    """True when the track's sideways spread is below `ratio` of its length (roll about it is loose)."""
    c = centers[np.all(np.isfinite(centers), axis=1)]
    if len(c) < 3:
        return True
    sv = np.linalg.svd(c - c.mean(0), compute_uv=False)
    return bool(sv[1] < ratio * sv[0])


STRAIGHT_TRACK_WARNING = ("the flight path is nearly straight, so GPS alone cannot fix the model's tilt "
                          "across the path; it may lean a few degrees until gravity from the IMU is applied")


def normalized_points(px: np.ndarray, K: CameraIntrinsics) -> np.ndarray:
    """Pixels -> undistorted normalised image coordinates."""
    x = (px - [K.cx, K.cy]) / [K.fx, K.fy]
    k1, k2 = (K.dist[0], K.dist[1]) if len(K.dist) >= 2 else (0.0, 0.0)
    return undistort_normalized(x, k1, k2) if (k1 or k2) else x


def relative_pose(x0: np.ndarray, x1: np.ndarray, focal_px: float
                  ) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Relative pose (R, unit t, inliers) with X1 = R X0 + t, from normalised points."""
    if len(x0) < 8:
        return None
    E, mask = cv2.findEssentialMat(x0, x1, np.eye(3), cv2.RANSAC, 0.999, 1.0 / focal_px)
    if E is None or mask is None:
        return None
    n, R, t, mask = cv2.recoverPose(E[:3], x0, x1, np.eye(3), mask=mask)
    return (R, t.ravel(), int(n)) if n >= 8 else None
