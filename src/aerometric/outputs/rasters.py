"""Raster products: DSM, orthophoto, trust / class / uncertainty layers, as GeoTIFF.

The local east/north frame is linear in latitude/longitude (see geo.LocalFrame),
so a regular metre grid is exactly a regular WGS84 degree grid. Rasters are
written as EPSG:4326 GeoTIFFs with tifffile, readable by QGIS/ArcGIS/GDAL.
Heights are in the same vertical datum as the georeferenced model.

  DSM      highest surface point per cell (mesh vertices and face centres),
           small holes filled from neighbours
  layers   trust (0/1/2), class and 1-sigma of the top surface point per cell
  ortho    each cell coloured from the most nadir keyframe that sees it (depth-map
           visibility test when depth maps are given); RGBA, alpha 0 where empty
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from aerometric.config import RasterConfig
from aerometric.depth.mono import AlignedDepth
from aerometric.geo import EARTH_RADIUS_M, LocalFrame
from aerometric.outputs.mesh_io import SurfaceAttributes
from aerometric.pose.types import Trajectory
from aerometric.types import DataError

NO_CLASS = 255


@dataclass(frozen=True)
class Grid:
    e0: float  # east of column 0 cell centre
    n0: float  # north of row 0 cell centre (rows go south)
    cell_m: float
    width: int
    height: int

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        E, N = np.meshgrid(self.e0 + self.cell_m * np.arange(self.width), self.n0 - self.cell_m * np.arange(self.height))
        return E, N

    def index(self, en: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = np.round((en[:, 0] - self.e0) / self.cell_m).astype(int)
        row = np.round((self.n0 - en[:, 1]) / self.cell_m).astype(int)
        return row, col


@dataclass(frozen=True)
class SurfaceRasters:
    grid: Grid
    dsm: np.ndarray  # HxW float32, NaN where empty
    trust: np.ndarray  # HxW uint8, 255 where empty
    classes: np.ndarray  # HxW uint8, 255 where empty/unlabelled
    sigma_m: np.ndarray  # HxW float32, NaN where empty
    filled: np.ndarray  # HxW bool, cells filled from neighbours rather than observed


def make_grid(vertices: np.ndarray, faces: np.ndarray, cfg: RasterConfig) -> Grid:
    if cfg.cell_m > 0:
        cell = cfg.cell_m
    else:
        e = vertices[faces[:, [0, 1, 2]]] - vertices[faces[:, [1, 2, 0]]]
        cell = 2.0 * float(np.median(np.linalg.norm(e, axis=2)))
    lo, hi = vertices[:, :2].min(0), vertices[:, :2].max(0)
    e0 = np.floor(lo[0] / cell) * cell
    n0 = np.ceil(hi[1] / cell) * cell
    return Grid(e0, n0, cell, int(np.ceil((hi[0] - e0) / cell)) + 1, int(np.ceil((n0 - lo[1]) / cell)) + 1)


def rasterize_surface(vertices: np.ndarray, faces: np.ndarray, attrs: SurfaceAttributes,
                      cfg: RasterConfig | None = None, grid: Grid | None = None) -> SurfaceRasters:
    cfg = cfg or RasterConfig()
    if len(vertices) == 0 or len(faces) == 0:
        raise DataError("rasters: the mesh is empty")
    grid = grid or make_grid(vertices, faces, cfg)
    # samples: vertices plus face centres (carry the attributes of the face's first vertex)
    cent = vertices[faces].mean(1)
    P = np.concatenate([vertices, cent])
    src = np.concatenate([np.arange(len(vertices)), faces[:, 0]])
    row, col = grid.index(P)
    ok = (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)
    P, src, row, col = P[ok], src[ok], row[ok], col[ok]
    flat = row * grid.width + col

    # top sample per cell: sort by (cell, -z) and take the first of each cell
    order = np.lexsort((-P[:, 2], flat))
    first = order[np.r_[True, flat[order][1:] != flat[order][:-1]]]
    size = grid.width * grid.height
    dsm = np.full(size, np.nan, np.float32)
    dsm[flat[first]] = P[first, 2]
    top = np.full(size, -1)
    top[flat[first]] = src[first]
    dsm, top = dsm.reshape(grid.height, grid.width), top.reshape(grid.height, grid.width)

    filled = np.zeros_like(dsm, bool)
    if cfg.fill_holes_px > 0:
        empty = np.isnan(dsm)
        dist, (ri, ci) = distance_transform_edt(empty, return_indices=True)
        fill = empty & (dist <= cfg.fill_holes_px)
        # only fill holes enclosed by data, not the area outside the mesh
        inside = cv2.morphologyEx((~empty).astype(np.uint8), cv2.MORPH_CLOSE,
                                  np.ones((2 * cfg.fill_holes_px + 1,) * 2, np.uint8)).astype(bool)
        fill &= inside
        dsm[fill] = dsm[ri[fill], ci[fill]]
        top[fill] = top[ri[fill], ci[fill]]
        filled = fill

    has = top >= 0

    def layer(values: np.ndarray | None, empty_value, dtype):
        out = np.full(top.shape, empty_value, dtype)
        if values is not None:
            out[has] = values[top[has]]
        return out

    return SurfaceRasters(grid, dsm, layer(attrs.trust, 255, np.uint8), layer(attrs.labels, NO_CLASS, np.uint8),
                          layer(attrs.sigma_m, np.nan, np.float32), filled)


# --------------------------------------------------------------------------- orthophoto

def best_view_colours(X: np.ndarray, traj: Trajectory, load_image: Callable[[int], np.ndarray],
                      depths: Sequence[AlignedDepth | None] | None = None, occlusion_tol: float = 0.05
                      ) -> tuple[np.ndarray, np.ndarray]:
    """RGB colour of each 3-D point from the most nadir keyframe that sees it. Returns (Nx3 uint8, seen)."""
    K = traj.intrinsics
    best_cos = np.full(len(X), -np.inf)
    colour = np.zeros((len(X), 3), np.uint8)
    for k in range(len(traj)):
        if not traj.valid[k]:
            continue
        p = traj.poses[k]
        Xc = X @ p.R_cw.T + p.t_cw
        z = Xc[:, 2]
        front = z > 1e-3
        u = np.where(front, K.fx * Xc[:, 0] / np.where(front, z, 1) + K.cx, -1)
        v = np.where(front, K.fy * Xc[:, 1] / np.where(front, z, 1) + K.cy, -1)
        vis = front & (u >= 0) & (u <= K.width - 1) & (v >= 0) & (v <= K.height - 1)
        if depths is not None and depths[k] is not None and depths[k].good:
            d = depths[k]
            s = 1.0 / d.scale_to_full
            dm = np.full(len(X), np.nan)
            dm[vis] = map_coordinates(d.depth.astype(float), [v[vis] * s, u[vis] * s], order=0, mode="nearest")
            vis &= np.isfinite(dm) & (np.abs(dm - z) <= occlusion_tol * z)
        ray = p.center - X
        cosz = ray[:, 2] / np.linalg.norm(ray, axis=1)  # 1 = camera straight above the cell
        better = vis & (cosz > best_cos)
        if not better.any():
            continue
        img = load_image(k)
        mx = u[better].astype(np.float32)[:, None]
        my = v[better].astype(np.float32)[:, None]
        # chunk to keep cv2.remap within its 32767-element map limit
        cols = []
        for i in range(0, len(mx), 30000):
            cols.append(cv2.remap(img, mx[i:i + 30000], my[i:i + 30000], cv2.INTER_LINEAR).reshape(-1, 3))
        colour[better] = np.concatenate(cols)
        best_cos[better] = cosz[better]
    seen = np.isfinite(best_cos)
    rgb = np.zeros((len(X), 3), np.uint8)
    rgb[seen] = colour[seen][:, ::-1]  # BGR -> RGB
    return rgb, seen


def orthophoto(rasters: SurfaceRasters, traj: Trajectory, load_image: Callable[[int], np.ndarray],
               depths: Sequence[AlignedDepth | None] | None = None, cfg: RasterConfig | None = None
               ) -> tuple[np.ndarray, tuple[str, ...]]:
    """RGBA uint8 orthophoto on the DSM grid. load_image(keyframe position) -> BGR image."""
    cfg = cfg or RasterConfig()
    g = rasters.grid
    E, N = g.centres()
    has = np.isfinite(rasters.dsm)
    X = np.column_stack([E[has], N[has], rasters.dsm[has]])
    rgb, seen = best_view_colours(X, traj, load_image, depths, cfg.occlusion_tol)
    out = np.zeros((g.height, g.width, 4), np.uint8)
    out[has, :3] = rgb
    out[has, 3] = np.where(seen, 255, 0)
    warnings = []
    if depths is None:
        warnings.append("orthophoto made without depth maps: surfaces hidden from a view may show what hides them")
    return out, tuple(warnings)


# --------------------------------------------------------------------------- GeoTIFF

def _geo_tags(grid: Grid, frame: LocalFrame, nodata: str | None) -> list[tuple]:
    dlat = np.degrees(grid.cell_m / EARTH_RADIUS_M)
    dlon = dlat / np.cos(np.radians(frame.lat0))
    lat_c, lon_c, _ = frame.to_lla(np.array([[grid.e0, grid.n0, 0.0]]))
    lon_edge, lat_edge = float(lon_c[0]) - dlon / 2, float(lat_c[0]) + dlat / 2
    keys = [1, 1, 0, 4,
            1024, 0, 1, 2,        # GTModelType: geographic
            1025, 0, 1, 1,        # GTRasterType: PixelIsArea
            2048, 0, 1, 4326,     # GeographicType: WGS 84
            2054, 0, 1, 9102]     # angular units: degree
    tags = [(33550, "d", 3, (dlon, dlat, 0.0), True),
            (33922, "d", 6, (0.0, 0.0, 0.0, lon_edge, lat_edge, 0.0), True),
            (34735, "H", len(keys), tuple(keys), True)]
    if nodata is not None:
        tags.append((42113, "s", 0, nodata, True))
    return tags


def write_geotiff(path: str | Path, data: np.ndarray, grid: Grid, frame: LocalFrame,
                  nodata: float | int | None = None, description: str = "") -> Path:
    import tifffile

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {"description": description} if description else {}
    if data.ndim == 3 and data.shape[2] == 4:
        kwargs.update(photometric="rgb", extrasamples=["unassalpha"])
    elif data.ndim == 3:
        kwargs.update(photometric="rgb")
    else:
        kwargs.update(photometric="minisblack")
    tifffile.imwrite(out, data, extratags=_geo_tags(grid, frame, None if nodata is None else str(nodata)), **kwargs)
    return out


def write_surface_rasters(folder: str | Path, rasters: SurfaceRasters, frame: LocalFrame,
                          ortho_rgba: np.ndarray | None = None, cfg: RasterConfig | None = None) -> dict[str, Path]:
    cfg = cfg or RasterConfig()
    d = Path(folder)
    g = rasters.grid
    dsm = np.where(np.isfinite(rasters.dsm), rasters.dsm, cfg.nodata).astype(np.float32)
    sig = np.where(np.isfinite(rasters.sigma_m), rasters.sigma_m, cfg.nodata).astype(np.float32)
    files = {
        "dsm": write_geotiff(d / "dsm.tif", dsm, g, frame, cfg.nodata, "Digital surface model, metres"),
        "trust": write_geotiff(d / "trust.tif", rasters.trust, g, frame, 255,
                               "Trust: 0 Measured, 1 Inferred, 2 Unknown, 255 no data"),
        "class": write_geotiff(d / "class.tif", rasters.classes, g, frame, NO_CLASS, "Semantic class, 255 none"),
        "sigma": write_geotiff(d / "sigma.tif", sig, g, frame, cfg.nodata, "1-sigma uncertainty, metres"),
    }
    if ortho_rgba is not None:
        files["ortho"] = write_geotiff(d / "ortho.tif", ortho_rgba, g, frame, None, "Orthophoto (RGBA)")
    return files
