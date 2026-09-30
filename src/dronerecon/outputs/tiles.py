"""3D Tiles 1.1 tileset and GeoJSON semantic/trust layers.

3D Tiles: the mesh is split into a flat quadtree of GLB tiles (faces go to the
tile holding their centroid) under a content-less root. Vertices are converted
through latitude/longitude/height exactly as every other product, then to ECEF
and into the exact WGS84 tangent frame at the model origin; the root transform
maps that frame to ECEF. GLB tiles carry the same trust/class attributes as the
main GLB. There is no level-of-detail simplification: viewers load the tiles
themselves once the root's geometric error is reached.

Heights are ellipsoidal as placed by georeferencing; if the model was matched to
a mean-sea-level DEM they carry that datum instead (the tileset notes which).

GeoJSON: the class and trust rasters are traced into polygons (with holes) in
WGS84, one feature per connected area, with its area in square metres.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from dronerecon.geo import LocalFrame, enu_basis, enu_to_ecef_matrix, lla_to_ecef
from dronerecon.outputs.mesh_io import SurfaceAttributes, write_glb
from dronerecon.outputs.rasters import Grid
from dronerecon.segmentation import Label
from dronerecon.trust import TrustLabel
from dronerecon.types import DataError


# --------------------------------------------------------------------------- 3D Tiles

def _tangent_coords(vertices: np.ndarray, frame: LocalFrame) -> tuple[np.ndarray, np.ndarray]:
    lat, lon, h = frame.to_lla(vertices)
    ecef = lla_to_ecef(lat, lon, h)
    origin = lla_to_ecef(np.array([frame.lat0]), np.array([frame.lon0]), np.array([frame.alt0]))[0]
    return (ecef - origin) @ enu_basis(frame.lat0, frame.lon0).T, np.column_stack([lat, lon, h])


def _region(lla: np.ndarray) -> list[float]:
    return [float(np.radians(lla[:, 1].min())), float(np.radians(lla[:, 0].min())),
            float(np.radians(lla[:, 1].max())), float(np.radians(lla[:, 0].max())),
            float(lla[:, 2].min()), float(lla[:, 2].max())]


def write_tileset(folder: str | Path, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray | None,
                  attrs: SurfaceAttributes, frame: LocalFrame, tile_m: float = 100.0,
                  height_datum: str = "ellipsoid (GPS)") -> Path:
    if len(faces) == 0:
        raise DataError("tiles: the mesh has no faces")
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    local, lla = _tangent_coords(vertices, frame)
    cent = vertices[faces].mean(1)[:, :2]
    lo = cent.min(0)
    key = np.floor((cent - lo) / tile_m).astype(int)
    children = []
    for tx, ty in np.unique(key, axis=0):
        sel = faces[(key[:, 0] == tx) & (key[:, 1] == ty)]
        used = np.unique(sel)
        remap = np.full(len(vertices), -1)
        remap[used] = np.arange(len(used))
        sub = SurfaceAttributes(*(None if a is None else a[used] for a in (attrs.trust, attrs.sigma_m, attrs.labels)))
        name = f"tile_{tx}_{ty}.glb"
        # normals: the model frame and the tangent frame differ by far less than a degree over a survey area
        write_glb(out / name, local[used], remap[sel], None if normals is None else normals[used],
                  sub, None, title=name[:-4])
        children.append({"boundingVolume": {"region": _region(lla[used])}, "geometricError": 0.0,
                         "content": {"uri": name}})
    root_error = float(np.hypot(*(vertices[:, :2].max(0) - vertices[:, :2].min(0))))
    tileset = {
        "asset": {"version": "1.1", "generator": "dronerecon",
                  "extras": {"height_datum": height_datum,
                             "trust_codes": {"0": "Measured", "1": "Inferred", "2": "Unknown"}}},
        "geometricError": root_error,
        "root": {"boundingVolume": {"region": _region(lla)}, "geometricError": root_error / 4,
                 "refine": "ADD", "transform": enu_to_ecef_matrix(frame.lat0, frame.lon0, frame.alt0).T.ravel().tolist(),
                 "children": children},
    }
    path = out / "tileset.json"
    path.write_text(json.dumps(tileset, indent=1), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- GeoJSON layers

CLASS_NAMES = {int(k): k.name.lower() for k in Label}
TRUST_NAMES = {int(k): k.name.capitalize() for k in TrustLabel}


def raster_to_geojson(layer: np.ndarray, grid: Grid, frame: LocalFrame, names: dict[int, str], property_name: str,
                      skip: tuple[int, ...] = (255,), simplify_cells: float = 0.5) -> dict:
    features = []
    for value in np.unique(layer):
        if int(value) in skip:
            continue
        mask = (layer == value).astype(np.uint8)
        contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        hier = hier[0]

        def ring(c: np.ndarray) -> list[list[float]]:
            c = cv2.approxPolyDP(c, simplify_cells, True).reshape(-1, 2).astype(float)
            en = np.column_stack([grid.e0 + c[:, 0] * grid.cell_m, grid.n0 - c[:, 1] * grid.cell_m, np.zeros(len(c))])
            lat, lon, _ = frame.to_lla(en)
            pts = [[round(float(a), 9), round(float(b), 9)] for a, b in zip(lon, lat)]
            return pts + [pts[0]]

        for i, c in enumerate(contours):
            if hier[i][3] != -1 or len(c) < 3:
                continue  # holes are attached to their outer ring below
            rings = [ring(c)]
            child = hier[i][2]
            while child != -1:
                if len(contours[child]) >= 3:
                    rings.append(ring(contours[child]))
                child = hier[child][0]
            area = float(cv2.contourArea(c)) * grid.cell_m ** 2
            area -= sum(float(cv2.contourArea(contours[j])) for j in range(len(contours)) if hier[j][3] == i) \
                * grid.cell_m ** 2
            if len(rings[0]) < 4:
                continue
            features.append({"type": "Feature",
                             "properties": {property_name: names.get(int(value), str(int(value))),
                                            "code": int(value), "area_m2": round(max(area, 0.0), 1)},
                             "geometry": {"type": "Polygon", "coordinates": rings}})
    return {"type": "FeatureCollection", "features": features}


def write_layers(folder: str | Path, classes: np.ndarray, trust: np.ndarray, grid: Grid,
                 frame: LocalFrame) -> dict[str, Path]:
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, layer, names, prop in (("classes", classes, CLASS_NAMES, "class"),
                                     ("trust", trust, TRUST_NAMES, "trust")):
        p = out / f"{name}.geojson"
        p.write_text(json.dumps(raster_to_geojson(layer, grid, frame, names, prop)), encoding="utf-8")
        files[name] = p
    return files
