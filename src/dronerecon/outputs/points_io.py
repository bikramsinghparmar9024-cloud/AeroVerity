"""LAS point clouds and an (untrained) Gaussian-splat PLY.

LAS 1.2, point format 2, written directly; geographic WGS84 coordinates
(longitude, latitude in degrees, ellipsoidal/GPS height in metres) with a GeoTIFF
GeoKeyDirectory VLR (EPSG:4326). Classification uses ASPRS codes; trust
(0 Measured, 1 Inferred, 2 Unknown) goes in the per-point User Data byte.
LAZ compression needs laspy with lazrs, which is not installed; asking for
.laz raises a clear error instead of writing a mislabelled file.

Gaussian splats: training needs gsplat (and a GPU), which is not installed. What
is written here is the standard 3DGS PLY layout SEEDED from the fused surface:
one flat Gaussian per point, lying along the surface, sized to the local point
spacing, coloured from the most nadir keyframe, with opacity from trust. It
opens in splat viewers as a quick visual model; it is NOT an optimised splat,
and the file says so.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from dronerecon.geo import LocalFrame
from dronerecon.segmentation import Label
from dronerecon.types import DataError

NO_LABEL = 255
ASPRS = {Label.OTHER: 1, Label.ROAD: 2, Label.BUILDING: 6, Label.VEGETATION: 5, Label.WATER: 9,
         Label.SKY: 1, Label.PERSON: 1, Label.VEHICLE: 1, Label.ANIMAL: 1}
SH_C0 = 0.28209479177387814
SPLAT_OPACITY = {0: 0.95, 1: 0.6, 2: 0.25}  # Measured, Inferred, Unknown


# --------------------------------------------------------------------------- LAS

def write_las(path: str | Path, points_enu: np.ndarray, frame: LocalFrame, rgb: np.ndarray | None = None,
              labels: np.ndarray | None = None, trust: np.ndarray | None = None) -> Path:
    out = Path(path)
    if out.suffix.lower() == ".laz":
        raise DataError("LAZ compression needs laspy with lazrs, which is not installed; write .las instead")
    n = len(points_enu)
    if n == 0:
        raise DataError("LAS: no points to write")
    for name, a in (("rgb", rgb), ("labels", labels), ("trust", trust)):
        if a is not None and len(a) != n:
            raise DataError(f"LAS: {name} has {len(a)} values for {n} points")

    lat, lon, h = frame.to_lla(points_enu)
    xyz = np.column_stack([lon, lat, h])
    scale = np.array([1e-7, 1e-7, 1e-3])
    offset = np.floor(xyz.min(0) * np.array([1, 1, 1]))
    q = np.round((xyz - offset) / scale).astype(np.int64)
    if np.abs(q).max() >= 2**31:
        raise DataError("LAS: coordinates exceed the 32-bit range at the chosen scale")

    rec = np.zeros(n, dtype=[("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"), ("intensity", "<u2"), ("flags", "u1"),
                             ("cls", "u1"), ("scan", "i1"), ("user", "u1"), ("src", "<u2"),
                             ("r", "<u2"), ("g", "<u2"), ("b", "<u2")])
    rec["X"], rec["Y"], rec["Z"] = q[:, 0], q[:, 1], q[:, 2]
    rec["flags"] = 0b00001001  # return 1 of 1
    if labels is not None:
        lut = np.ones(256, np.uint8)
        for k, v in ASPRS.items():
            lut[int(k)] = v
        rec["cls"] = lut[labels]
    else:
        rec["cls"] = 1
    if trust is not None:
        rec["user"] = trust
    if rgb is not None:
        rgb16 = rgb.astype(np.uint16) * 257  # 8-bit -> 16-bit full range
        rec["r"], rec["g"], rec["b"] = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]

    keys = [1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326]
    vlr_body = struct.pack(f"<{len(keys)}H", *keys)
    vlr = struct.pack("<H16sHH32s", 0, b"LASF_Projection", 34735, len(vlr_body), b"GeoKeyDirectory") + vlr_body
    header_size, rec_len = 227, 26
    offset_to_points = header_size + len(vlr)
    by_return = [n, 0, 0, 0, 0]
    header = struct.pack(
        "<4sHH16sBB32s32sHHHIIBHI5I3d3d6d",
        b"LASF", 0, 0, b"\0" * 16, 1, 2, b"dronerecon".ljust(32, b"\0"),
        b"dronerecon (trust in User Data)".ljust(32, b"\0"), 1, 2026, header_size, offset_to_points, 1, 2,
        rec_len, n, *by_return, *scale, *offset,
        xyz[:, 0].max(), xyz[:, 0].min(), xyz[:, 1].max(), xyz[:, 1].min(), xyz[:, 2].max(), xyz[:, 2].min())
    if len(header) != header_size:
        raise AssertionError(f"LAS header is {len(header)} bytes, expected {header_size}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(header)
        f.write(vlr)
        f.write(rec.tobytes())
    return out


# --------------------------------------------------------------------------- Gaussian splat seed

def _quat_z_to(normals: np.ndarray) -> np.ndarray:
    """Unit quaternions (w, x, y, z) rotating +z onto each normal."""
    n = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(np.broadcast_to(z, n.shape), n)
    w = 1.0 + n[:, 2]
    q = np.column_stack([w, axis])
    flip = w < 1e-9  # normal points straight down: rotate 180 degrees about x
    q[flip] = [0.0, 1.0, 0.0, 0.0]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def write_splat_seed(path: str | Path, points: np.ndarray, normals: np.ndarray, rgb: np.ndarray,
                     trust: np.ndarray | None = None, frame: LocalFrame | None = None,
                     flatness: float = 0.1) -> Path:
    n = len(points)
    if n < 2:
        raise DataError("splat: need at least two points")
    if len(normals) != n or len(rgb) != n or (trust is not None and len(trust) != n):
        raise DataError("splat: points, normals, colours and trust must have the same length")
    d, _ = cKDTree(points).query(points, k=min(4, n))
    spacing = np.maximum(d[:, 1:].mean(1), 1e-3)
    s = np.log(np.column_stack([spacing, spacing, spacing * flatness]))
    alpha = np.full(n, 0.9) if trust is None else np.vectorize(SPLAT_OPACITY.get)(trust.astype(int), 0.25)
    logit = np.log(alpha / (1 - alpha))
    f_dc = (rgb.astype(np.float64) / 255.0 - 0.5) / SH_C0
    q = _quat_z_to(normals)

    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.column_stack([points, np.zeros((n, 3)), f_dc, logit, s, q]).astype("<f4")
    lines = ["ply", "format binary_little_endian 1.0",
             "comment UNTRAINED Gaussian-splat seed from the fused surface (not optimised with gsplat)",
             "comment coordinates local ENU metres (x=east, y=north, z=up)"]
    if frame is not None:
        lines.append(f"comment origin_wgs84 lat={frame.lat0:.9f} lon={frame.lon0:.9f} alt_m={frame.alt0:.3f}")
    lines += [f"element vertex {n}"] + [f"property float {k}" for k in names] + ["end_header"]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        f.write(data.tobytes())
    return out
