"""Camera intrinsics from JSON.

Accepted layouts:
  {"width", "height", "fx", "fy", "cx", "cy", "dist": [...]}
  {"image_width", "image_height", "camera_matrix": [[...]], "dist_coeffs": [...]}  (OpenCV style)
  {"hfov_deg": 84}  (rough prior; image size taken from the video)
A calibration made at a different resolution with the same aspect ratio is scaled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dronerecon.types import CameraIntrinsics, DataError


def _from_dict(d: dict[str, Any], width: int, height: int, name: str) -> CameraIntrinsics:
    if "hfov_deg" in d:
        return CameraIntrinsics.from_fov(width, height, float(d["hfov_deg"]))
    if "camera_matrix" in d:
        K = np.asarray(d["camera_matrix"], dtype=float)
        if K.shape != (3, 3):
            raise DataError(f"{name}: camera_matrix must be 3x3")
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        w, h = d.get("image_width", width), d.get("image_height", height)
        dist = d.get("dist_coeffs", [])
    else:
        try:
            fx, fy, cx, cy = (float(d[k]) for k in ("fx", "fy", "cx", "cy"))
        except KeyError as exc:
            raise DataError(f"{name}: missing intrinsics field {exc}") from None
        w, h = d.get("width", width), d.get("height", height)
        dist = d.get("dist", [])
    dist = tuple(float(v) for v in np.ravel(dist))
    if any(dist) and len(dist) not in (4, 5):
        raise DataError(f"{name}: expected 4 or 5 OpenCV distortion terms, got {len(dist)}")
    dist = dist if any(dist) else ()

    if (w, h) != (width, height):
        if abs(w / h - width / height) > 1e-3:
            raise DataError(f"{name}: calibrated at {w}x{h}, which has a different aspect "
                            f"ratio from the video ({width}x{height}); cropping is not supported")
        s = width / w
        fx, fy, cx, cy = fx * s, fy * s, cx * s, cy * s
    return CameraIntrinsics(width, height, fx, fy, cx, cy,
                            model="opencv" if dist else "pinhole", dist=dist, source="provided")


def load_intrinsics(path: str | Path, width: int, height: int) -> CameraIntrinsics:
    """Load intrinsics for a video of the given size."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DataError(f"intrinsics file not found: {p}") from None
    except json.JSONDecodeError as exc:
        raise DataError(f"{p.name}: invalid JSON ({exc})") from None
    if not isinstance(data, dict):
        raise DataError(f"{p.name}: expected a JSON object")
    return _from_dict(data, width, height, p.name)
