"""Horizontal registration against a local, georeferenced satellite/aerial image.

Offline inputs (WGS84 latitude/longitude):
  * GeoTIFF (RGB or grey) with pixel-scale/tie-point tags
  * PNG/JPEG/TIFF with a world file (.pgw/.jgw/.tfw/.wld) in degrees

For a spread of keyframes, the frame is projected onto the ground plane using
its georeferenced pose, giving a small orthophoto on an east/north grid. The
satellite image is resampled onto the same grid (plus a search margin), the two
are feature-matched, and a robust 2-D fit gives how far the model is off. The
per-keyframe shifts must agree; their median is blended with the existing
position using both uncertainties, so imagery improves GPS but cannot replace it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from aerometric.config import GeorefConfig, MatchingConfig
from aerometric.geo import LocalFrame
from aerometric.georef.dem import _check_geographic, read_geotiff
from aerometric.georef.similarity import GeoRegistration
from aerometric.matching.features import make_matching
from aerometric.pose.types import Similarity
from aerometric.quality.coverage import ground_footprint
from aerometric.types import DataError

WORLD_FILE_SUFFIXES = (".pgw", ".jgw", ".tfw", ".wld", ".pngw", ".jpgw", ".tifw")


@dataclass(frozen=True)
class GeoImage:
    image: np.ndarray  # HxWx3 uint8 BGR
    lat_top: float  # latitude of row 0 pixel centres
    lon_left: float  # longitude of column 0 pixel centres
    dlat: float  # degrees per row (positive, rows go south)
    dlon: float

    def gsd_m(self, frame: LocalFrame) -> float:
        return float(np.radians(self.dlat) * 6_378_137.0)

    def resample_enu(self, frame: LocalFrame, e0: float, n1: float, res: float, w: int, h: int) -> np.ndarray:
        """Resample onto an ENU grid whose top-left pixel centre is (e0, n1), rows going south."""
        E, N = np.meshgrid(e0 + res * np.arange(w), n1 - res * np.arange(h))
        lat, lon, _ = frame.to_lla(np.column_stack([E.ravel(), N.ravel(), np.zeros(E.size)]))
        mx = ((lon - self.lon_left) / self.dlon).reshape(h, w).astype(np.float32)
        my = ((self.lat_top - lat) / self.dlat).reshape(h, w).astype(np.float32)
        return cv2.remap(self.image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def load_geoimage(path: str | Path) -> GeoImage:
    p = Path(path)
    world = next((p.with_suffix(s) for s in WORLD_FILE_SUFFIXES if p.with_suffix(s).is_file()), None)
    if world is not None:
        a, d, b, e, c, f = (float(v) for v in world.read_text().split()[:6])
        if abs(b) > 1e-12 or abs(d) > 1e-12:
            raise DataError(f"{world.name}: rotated world files are not supported")
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise DataError(f"could not read image {p}")
        _check_geographic(world.name, c, f, a)
        return GeoImage(img, f, c, -e, a)
    if p.suffix.lower() in (".tif", ".tiff"):
        arr, lat0, lon0, dlat, dlon, _ = read_geotiff(p)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            arr = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        return GeoImage(arr, lat0, lon0, dlat, dlon)
    raise DataError(f"{p.name}: no georeferencing found (GeoTIFF tags or a world file such as .pgw)")


# --------------------------------------------------------------------------- ortho projection

def ortho_keyframe(image: np.ndarray, traj, k: int, z0: float, e0: float, n1: float, res: float,
                   w: int, h: int) -> np.ndarray:
    """Project keyframe k onto the plane z = z0, sampled on an ENU grid."""
    K, p = traj.intrinsics, traj.poses[k]
    E, N = np.meshgrid(e0 + res * np.arange(w), n1 - res * np.arange(h))
    X = np.column_stack([E.ravel(), N.ravel(), np.full(E.size, z0)])
    Xc = X @ p.R_cw.T + p.t_cw
    front = Xc[:, 2] > 0.1
    u = np.where(front, K.fx * Xc[:, 0] / np.where(front, Xc[:, 2], 1) + K.cx, -1)
    v = np.where(front, K.fy * Xc[:, 1] / np.where(front, Xc[:, 2], 1) + K.cy, -1)
    return cv2.remap(image, u.reshape(h, w).astype(np.float32), v.reshape(h, w).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# --------------------------------------------------------------------------- registration

@dataclass(frozen=True)
class SatelliteRegistration:
    registration: GeoRegistration
    measured_shift_en: np.ndarray  # median shift the imagery asks for (metres east, north)
    applied_shift_en: np.ndarray  # after blending with the existing position uncertainty
    frame_shifts_en: np.ndarray  # Fx2, one per keyframe that matched
    frames_matched: int
    spread_m: float
    warnings: tuple[str, ...] = ()


def _frame_shift(ortho: np.ndarray, sat: np.ndarray, off_px: float, res: float, cfg: GeorefConfig
                 ) -> np.ndarray | None:
    ext, mat = make_matching(MatchingConfig(backend="opencv", max_dim=max(sat.shape[:2]), ratio=0.85))
    valid = cv2.cvtColor(ortho, cv2.COLOR_BGR2GRAY) > 0
    fo = ext.extract(ortho, ~cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool))
    fs = ext.extract(sat)
    m = mat.match(fo, fs)
    if len(m) < cfg.sat_min_inliers:
        return None
    src = fo.keypoints[m.idx0] + off_px  # ortho pixels expressed in satellite-grid pixels
    dst = fs.keypoints[m.idx1]
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or int(inl.sum()) < cfg.sat_min_inliers:
        return None
    c = np.array([ortho.shape[1] / 2, ortho.shape[0] / 2]) + off_px
    d = M[:, :2] @ c + M[:, 2] - c  # pixel shift at the frame centre
    return np.array([d[0], -d[1]]) * res  # rows go south


def register_to_satellite(reg: GeoRegistration, sat: GeoImage, load_image: Callable[[int], np.ndarray],
                          cfg: GeorefConfig | None = None) -> SatelliteRegistration:
    """load_image: keyframe position (0..N-1) -> BGR image."""
    cfg = cfg or GeorefConfig()
    traj = reg.trajectory
    z0 = float(np.median(reg.points[:, 2])) if len(reg.points) else float(np.min(traj.centers[:, 2]) - 100)
    res = max(sat.gsd_m(reg.frame), 0.05)
    candidates = [k for k in range(len(traj)) if traj.valid[k]]
    picks = [candidates[i] for i in np.linspace(0, len(candidates) - 1,
                                                min(cfg.sat_max_frames, len(candidates))).round().astype(int)]
    margin = int(np.ceil(cfg.sat_search_m / res))
    shifts = []
    for k in dict.fromkeys(picks):
        foot = ground_footprint(traj, k, z0)
        if foot is None:
            continue
        e0, n0 = foot.min(0)
        e1, n1 = foot.max(0)
        w, h = int((e1 - e0) / res) + 1, int((n1 - n0) / res) + 1
        ortho = ortho_keyframe(load_image(k), traj, k, z0, e0, n1, res, w, h)
        satc = sat.resample_enu(reg.frame, e0 - margin * res, n1 + margin * res, res, w + 2 * margin, h + 2 * margin)
        if not satc.any():
            continue
        s = _frame_shift(ortho, satc, margin, res, cfg)
        if s is not None and np.all(np.abs(s) <= cfg.sat_search_m):
            shifts.append(s)

    warnings: list[str] = []
    none = np.zeros(2)
    if len(shifts) < 2:
        warnings.append("the satellite image could not be matched reliably to the flight "
                        f"({len(shifts)} keyframe(s) matched); horizontal position left unchanged")
        return SatelliteRegistration(reg, none, none, np.array(shifts).reshape(-1, 2), len(shifts), float("nan"),
                                     tuple(warnings))
    S = np.array(shifts)
    med = np.median(S, axis=0)
    spread = float(1.4826 * np.median(np.linalg.norm(S - med, axis=1)))
    if spread > cfg.sat_max_spread_m:
        warnings.append(f"keyframes disagree on the satellite offset (spread {spread:.1f} m); "
                        "horizontal position left unchanged")
        return SatelliteRegistration(reg, med, none, S, len(S), spread, tuple(warnings))

    # blend: the model's own absolute position is known to about its GPS RMS
    var_model, var_sat = max(reg.gps_rms_m, 0.1) ** 2, cfg.sat_sigma_m ** 2 + spread ** 2 / len(S)
    applied = med * var_model / (var_model + var_sat)
    corr = Similarity(1.0, np.eye(3), np.array([applied[0], applied[1], 0.0]))
    t = reg.transform
    new = replace(reg, transform=Similarity(t.s, t.R, t.t + corr.t),
                  trajectory=replace(traj, poses=tuple(corr.apply_pose(p) for p in traj.poses)),
                  points=corr.apply_points(reg.points) if len(reg.points) else reg.points,
                  anchors=reg.anchors + ("satellite",))
    if np.linalg.norm(med) > 3 * max(reg.gps_rms_m, cfg.sat_sigma_m):
        warnings.append(f"satellite imagery places the model {np.linalg.norm(med):.1f} m from where GPS put it; "
                        "check the imagery's georeferencing")
    new = replace(new, warnings=reg.warnings + tuple(warnings))
    return SatelliteRegistration(new, med, applied, S, len(S), spread, tuple(warnings))
