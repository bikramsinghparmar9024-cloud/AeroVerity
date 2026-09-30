"""Shared data structures passed between pipeline stages.

Clock convention: every time series stores seconds on its *own* source clock.
Telemetry-derived series (GPS, IMU, baro) share the telemetry clock; video
frames use the video clock. The sync stage estimates the offset between them;
nothing here assumes they are aligned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

GpsSource = Literal["gnss", "rtk", "ppk"]
IntrinsicsSource = Literal["provided", "estimated", "default"]
CameraModel = Literal["pinhole", "opencv"]


class DataError(ValueError):
    """Raised when input data violates a stage contract."""


def _as_array(name: str, value: object, ndim: int, cols: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != ndim:
        raise DataError(f"{name}: expected {ndim}-D array, got shape {arr.shape}")
    if cols is not None and arr.shape[1] != cols:
        raise DataError(f"{name}: expected {cols} columns, got shape {arr.shape}")
    arr.setflags(write=False)
    return arr


def _check_time(name: str, t: np.ndarray) -> None:
    if t.size == 0:
        raise DataError(f"{name}: time series is empty")
    if not np.all(np.isfinite(t)):
        raise DataError(f"{name}: time contains NaN/inf")
    if np.any(np.diff(t) <= 0):
        raise DataError(f"{name}: timestamps must be strictly increasing")


def _check_len(name: str, n: int, **arrays: np.ndarray | None) -> None:
    for key, arr in arrays.items():
        if arr is not None and len(arr) != n:
            raise DataError(f"{name}.{key}: length {len(arr)} != {n} timestamps")


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str = "unknown"
    creation_time_utc: str | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise DataError(f"VideoInfo: invalid resolution {self.width}x{self.height}")
        if not self.fps > 0:
            raise DataError(f"VideoInfo: fps must be positive, got {self.fps}")
        if self.frame_count <= 0:
            raise DataError(f"VideoInfo: frame_count must be positive, got {self.frame_count}")

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True)
class Frame:
    index: int
    t_video_s: float
    image: np.ndarray  # HxWx3 uint8, BGR


@dataclass(frozen=True)
class GpsTrack:
    """Positions in WGS84; alt_m is ellipsoidal or MSL as reported by the source."""

    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt_m: np.ndarray
    source: GpsSource = "gnss"
    h_acc_m: np.ndarray | None = None
    v_acc_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("t", "lat", "lon", "alt_m", "h_acc_m", "v_acc_m"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _as_array(f"GpsTrack.{name}", value, 1))
        _check_time("GpsTrack", self.t)
        _check_len("GpsTrack", len(self.t), lat=self.lat, lon=self.lon, alt_m=self.alt_m,
                   h_acc_m=self.h_acc_m, v_acc_m=self.v_acc_m)
        if np.any(np.abs(self.lat) > 90) or np.any(np.abs(self.lon) > 180):
            raise DataError("GpsTrack: latitude/longitude out of range")

    def __len__(self) -> int:
        return len(self.t)


@dataclass(frozen=True)
class AttitudeSeries:
    """Gimbal/camera attitude in degrees (yaw, pitch, roll)."""

    t: np.ndarray
    ypr_deg: np.ndarray  # Nx3

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _as_array("AttitudeSeries.t", self.t, 1))
        object.__setattr__(self, "ypr_deg", _as_array("AttitudeSeries.ypr_deg", self.ypr_deg, 2, 3))
        _check_time("AttitudeSeries", self.t)
        _check_len("AttitudeSeries", len(self.t), ypr_deg=self.ypr_deg)


@dataclass(frozen=True)
class ImuSeries:
    t: np.ndarray
    gyro_rad_s: np.ndarray  # Nx3
    accel_m_s2: np.ndarray | None = None  # Nx3

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _as_array("ImuSeries.t", self.t, 1))
        object.__setattr__(self, "gyro_rad_s", _as_array("ImuSeries.gyro_rad_s", self.gyro_rad_s, 2, 3))
        if self.accel_m_s2 is not None:
            object.__setattr__(self, "accel_m_s2",
                               _as_array("ImuSeries.accel_m_s2", self.accel_m_s2, 2, 3))
        _check_time("ImuSeries", self.t)
        _check_len("ImuSeries", len(self.t), gyro_rad_s=self.gyro_rad_s, accel_m_s2=self.accel_m_s2)


@dataclass(frozen=True)
class BaroSeries:
    t: np.ndarray
    alt_m: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _as_array("BaroSeries.t", self.t, 1))
        object.__setattr__(self, "alt_m", _as_array("BaroSeries.alt_m", self.alt_m, 1))
        _check_time("BaroSeries", self.t)
        _check_len("BaroSeries", len(self.t), alt_m=self.alt_m)


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: CameraModel = "pinhole"
    dist: tuple[float, ...] = ()  # OpenCV order: k1, k2, p1, p2[, k3]
    source: IntrinsicsSource = "provided"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise DataError("CameraIntrinsics: invalid image size")
        if self.fx <= 0 or self.fy <= 0:
            raise DataError("CameraIntrinsics: focal lengths must be positive")
        if self.model == "pinhole" and self.dist:
            raise DataError("CameraIntrinsics: pinhole model takes no distortion terms")

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float) -> "CameraIntrinsics":
        """Rough default when nothing better is known; refined by self-calibration."""
        if not 0 < hfov_deg < 180:
            raise DataError(f"CameraIntrinsics: hfov must be in (0, 180), got {hfov_deg}")
        f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
        return cls(width, height, f, f, width / 2, height / 2, source="default")


@dataclass(frozen=True)
class FlightRecord:
    """Everything ingestion hands to the sync stage."""

    video: VideoInfo
    gps: GpsTrack
    intrinsics: CameraIntrinsics
    attitude: AttitudeSeries | None = None
    imu: ImuSeries | None = None
    baro: BaroSeries | None = None
    rtk: GpsTrack | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if (self.intrinsics.width, self.intrinsics.height) != (self.video.width, self.video.height):
            raise DataError(
                f"FlightRecord: intrinsics size {self.intrinsics.width}x{self.intrinsics.height} "
                f"does not match video {self.video.width}x{self.video.height}"
            )
        if self.rtk is not None and self.rtk.source == "gnss":
            raise DataError("FlightRecord: rtk track must have source 'rtk' or 'ppk'")
