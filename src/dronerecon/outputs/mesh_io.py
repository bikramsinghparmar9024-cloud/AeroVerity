"""GLB and PLY writers that keep trust and semantics with the geometry.

Both formats are written directly (no extra dependency) so every vertex carries:
  colour   trust colour by default (green Measured, amber Inferred, red Unknown)
           or class colour; any viewer shows it
  _TRUST   0 Measured / 1 Inferred / 2 Unknown
  _SIGMA   effective 1-sigma in metres (NaN where unknown)
  _CLASS   semantic label (255 = unlabelled)
Coordinates are metres in the local east/north/up frame; its WGS84 origin is
stored in the file (glTF asset.extras, PLY comments). glTF is Y-up, so the GLB
maps (E, N, U) -> (x=E, y=U, z=-N).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from dronerecon.geo import LocalFrame
from dronerecon.segmentation import Label
from dronerecon.types import DataError

TRUST_COLORS = np.array([[46, 160, 90], [230, 160, 40], [200, 60, 60]], np.uint8)
NO_LABEL = 255
CLASS_COLORS = {
    Label.OTHER: (150, 150, 150), Label.ROAD: (90, 90, 100), Label.BUILDING: (200, 120, 90),
    Label.VEGETATION: (70, 150, 60), Label.WATER: (60, 110, 200), Label.SKY: (170, 210, 240),
    Label.PERSON: (220, 60, 160), Label.VEHICLE: (240, 200, 40), Label.ANIMAL: (160, 90, 40),
}
ColorBy = Literal["trust", "class"]


@dataclass(frozen=True)
class SurfaceAttributes:
    trust: np.ndarray | None = None  # V uint8
    sigma_m: np.ndarray | None = None  # V float
    labels: np.ndarray | None = None  # V uint8


def vertex_colors(attrs: SurfaceAttributes, n: int, color_by: ColorBy = "trust") -> np.ndarray:
    if color_by == "trust" and attrs.trust is not None:
        return TRUST_COLORS[np.clip(attrs.trust, 0, 2)]
    if color_by == "class" and attrs.labels is not None:
        lut = np.full((256, 3), 110, np.uint8)
        for k, c in CLASS_COLORS.items():
            lut[int(k)] = c
        return lut[attrs.labels]
    return np.full((n, 3), 180, np.uint8)


def _check(vertices: np.ndarray, faces: np.ndarray | None, attrs: SurfaceAttributes) -> None:
    n = len(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise DataError("export: vertices must be Nx3")
    if faces is not None and len(faces) and (faces.min() < 0 or faces.max() >= n):
        raise DataError("export: face index out of range")
    for name in ("trust", "sigma_m", "labels"):
        a = getattr(attrs, name)
        if a is not None and len(a) != n:
            raise DataError(f"export: {name} has {len(a)} values for {n} vertices")


# --------------------------------------------------------------------------- GLB

def write_glb(path: str | Path, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray | None = None,
              attrs: SurfaceAttributes = SurfaceAttributes(), frame: LocalFrame | None = None,
              color_by: ColorBy = "trust", title: str = "") -> Path:
    _check(vertices, faces, attrs)
    if len(faces) == 0:
        raise DataError("export: a GLB needs at least one face")
    n = len(vertices)
    to_gltf = lambda a: np.column_stack([a[:, 0], a[:, 2], -a[:, 1]]).astype(np.float32)  # noqa: E731
    pos = to_gltf(vertices)
    arrays: list[tuple[str, np.ndarray, int, str, bool]] = [("POSITION", pos, 5126, "VEC3", False)]
    if normals is not None:
        nrm = to_gltf(normals)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        arrays.append(("NORMAL", nrm, 5126, "VEC3", False))
    rgba = np.column_stack([vertex_colors(attrs, n, color_by), np.full(n, 255, np.uint8)]).astype(np.uint8)
    arrays.append(("COLOR_0", rgba, 5121, "VEC4", True))
    if attrs.trust is not None:
        arrays.append(("_TRUST", attrs.trust.astype(np.float32), 5126, "SCALAR", False))
    if attrs.sigma_m is not None:
        arrays.append(("_SIGMA", attrs.sigma_m.astype(np.float32), 5126, "SCALAR", False))
    if attrs.labels is not None:
        arrays.append(("_CLASS", attrs.labels.astype(np.float32), 5126, "SCALAR", False))

    blob = bytearray()
    views, accessors, attributes = [], [], {}

    def add(data: np.ndarray, comp: int, typ: str, normalized: bool, target: int | None, name: str | None):
        while len(blob) % 4:
            blob.append(0)
        raw = np.ascontiguousarray(data).tobytes()
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(raw)}
        if target:
            view["target"] = target
        views.append(view)
        blob.extend(raw)
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": int(len(data)), "type": typ}
        if normalized:
            acc["normalized"] = True
        if name == "POSITION":
            acc["min"] = pos.min(0).tolist()
            acc["max"] = pos.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    for name, data, comp, typ, normed in arrays:
        attributes[name] = add(data, comp, typ, normed, 34962, name)
    idx = add(faces.astype(np.uint32).ravel(), 5125, "SCALAR", False, 34963, None)
    while len(blob) % 4:
        blob.append(0)

    extras = {"coordinates": "local ENU metres mapped to glTF as x=E, y=U, z=-N",
              "trust_codes": {"0": "Measured", "1": "Inferred", "2": "Unknown"}, "color_by": color_by}
    if frame is not None:
        extras["origin_wgs84"] = {"lat": frame.lat0, "lon": frame.lon0, "alt_m": frame.alt0}
    gltf = {
        "asset": {"version": "2.0", "generator": "dronerecon", "extras": extras},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": title or "reconstruction"}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": idx, "mode": 4}]}],
        "buffers": [{"byteLength": len(blob)}], "bufferViews": views, "accessors": accessors,
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with out.open("wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack("<II", len(blob), 0x004E4942))
        f.write(bytes(blob))
    return out


# --------------------------------------------------------------------------- PLY

def write_ply(path: str | Path, vertices: np.ndarray, faces: np.ndarray | None = None,
              normals: np.ndarray | None = None, attrs: SurfaceAttributes = SurfaceAttributes(),
              frame: LocalFrame | None = None, color_by: ColorBy = "trust") -> Path:
    """Binary little-endian PLY; point cloud when `faces` is None."""
    _check(vertices, faces, attrs)
    n = len(vertices)
    fields: list[tuple[str, str, np.ndarray]] = [
        ("x", "double", vertices[:, 0]), ("y", "double", vertices[:, 1]), ("z", "double", vertices[:, 2])]
    if normals is not None:
        fields += [("nx", "float", normals[:, 0]), ("ny", "float", normals[:, 1]), ("nz", "float", normals[:, 2])]
    rgb = vertex_colors(attrs, n, color_by)
    fields += [("red", "uchar", rgb[:, 0]), ("green", "uchar", rgb[:, 1]), ("blue", "uchar", rgb[:, 2])]
    if attrs.trust is not None:
        fields.append(("trust", "uchar", attrs.trust))
    if attrs.sigma_m is not None:
        fields.append(("sigma_m", "float", attrs.sigma_m))
    if attrs.labels is not None:
        fields.append(("class", "uchar", attrs.labels))

    np_type = {"double": "<f8", "float": "<f4", "uchar": "u1"}
    rec = np.empty(n, dtype=[(name, np_type[t]) for name, t, _ in fields])
    for name, _, data in fields:
        rec[name] = data

    lines = ["ply", "format binary_little_endian 1.0", "comment generator dronerecon",
             "comment coordinates local ENU metres (x=east, y=north, z=up)",
             "comment trust 0=Measured 1=Inferred 2=Unknown; class 255=unlabelled"]
    if frame is not None:
        lines.append(f"comment origin_wgs84 lat={frame.lat0:.9f} lon={frame.lon0:.9f} alt_m={frame.alt0:.3f}")
    lines.append(f"element vertex {n}")
    lines += [f"property {t} {name}" for name, t, _ in fields]
    if faces is not None:
        lines += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    lines.append("end_header")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        f.write(rec.tobytes())
        if faces is not None:
            fr = np.empty(len(faces), dtype=[("n", "u1"), ("v", "<i4", (3,))])
            fr["n"] = 3
            fr["v"] = faces
            f.write(fr.tobytes())
    return out
