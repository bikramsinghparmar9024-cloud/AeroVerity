"""Registration against a digital elevation model (DEM), fully offline.

Supported inputs (WGS84 latitude/longitude grids, heights in metres):
  * ESRI ASCII grid (.asc)
  * GeoTIFF with ModelPixelScale/ModelTiepoint tags (read with tifffile)
Projected DEMs (e.g. UTM) need reprojection to WGS84 first; no projection
library is bundled.

Registration nudges the georeferenced model so its sparse points sit on the
terrain: a vertical offset, a small tilt, and a horizontal shift when the
terrain has enough relief for a shift to be observable. Existing estimates act
as priors so a coarse DEM cannot override good GPS or gravity. Points on roofs
or trees sit above a bare-earth DEM; a robust loss keeps them from biasing the fit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from dronerecon.config import GeorefConfig
from dronerecon.geo import LocalFrame
from dronerecon.georef.similarity import GeoRegistration
from dronerecon.pose.types import Similarity
from dronerecon.types import DataError


@dataclass(frozen=True)
class ElevationModel:
    """Heights on a regular latitude/longitude grid; row 0 is the northernmost row."""

    heights: np.ndarray  # HxW float, NaN = no data
    lat_top: float  # latitude of row 0 cell centres
    lon_left: float  # longitude of column 0 cell centres
    dlat: float  # degrees per row (positive, rows go south)
    dlon: float  # degrees per column

    def height_at_lla(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        r = (self.lat_top - np.asarray(lat)) / self.dlat
        c = (np.asarray(lon) - self.lon_left) / self.dlon
        h, w = self.heights.shape
        inside = (r >= 0) & (r <= h - 1) & (c >= 0) & (c <= w - 1)
        out = np.full(r.shape, np.nan)
        if inside.any():
            out[inside] = map_coordinates(self.heights, [r[inside], c[inside]], order=1, mode="nearest")
        return out

    def height_at_enu(self, frame: LocalFrame, xy: np.ndarray) -> np.ndarray:
        lat, lon, _ = frame.to_lla(np.column_stack([xy, np.zeros(len(xy))]))
        return self.height_at_lla(lat, lon)


# --------------------------------------------------------------------------- loaders

def load_esri_ascii(path: str | Path) -> ElevationModel:
    p = Path(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    header: dict[str, float] = {}
    i = 0
    while i < len(lines) and lines[i].split() and re.fullmatch(r"[A-Za-z_]+", lines[i].split()[0]):
        k, v = lines[i].split()[:2]
        header[k.lower()] = float(v)
        i += 1
    try:
        ncols, nrows, cell = int(header["ncols"]), int(header["nrows"]), header["cellsize"]
    except KeyError as exc:
        raise DataError(f"{p.name}: missing ESRI ASCII header field {exc}") from None
    data = np.array([float(v) for line in lines[i:] for v in line.split()], float)
    if data.size != ncols * nrows:
        raise DataError(f"{p.name}: expected {ncols * nrows} values, found {data.size}")
    grid = data.reshape(nrows, ncols)
    if "nodata_value" in header:
        grid[grid == header["nodata_value"]] = np.nan
    if "xllcenter" in header:
        x0, y0 = header["xllcenter"], header["yllcenter"]
    else:
        x0, y0 = header["xllcorner"] + cell / 2, header["yllcorner"] + cell / 2
    _check_geographic(p.name, x0, y0, cell)
    return ElevationModel(grid, y0 + (nrows - 1) * cell, x0, cell, cell)


def load_geotiff(path: str | Path) -> ElevationModel:
    grid, lat0, lon0, dlat, dlon, nodata = read_geotiff(path)
    grid = grid.astype(float)
    if nodata is not None:
        grid[grid == nodata] = np.nan
    return ElevationModel(grid, lat0, lon0, dlat, dlon)


def read_geotiff(path: str | Path) -> tuple[np.ndarray, float, float, float, float, float | None]:
    """Raw pixels plus (lat of row 0 centres, lon of column 0 centres, dlat, dlon, nodata)."""
    import tifffile

    p = Path(path)
    with tifffile.TiffFile(p) as tif:
        page = tif.pages[0]
        tags = {t.code: t.value for t in page.tags.values()}
        grid = page.asarray()
    if 33550 not in tags or 33922 not in tags:
        raise DataError(f"{p.name}: no GeoTIFF pixel-scale/tie-point tags")
    sx, sy = tags[33550][:2]
    i, j, _, x, y, _ = tags[33922][:6]
    keys = tags.get(34735)
    if keys is not None:
        geo = dict(zip(keys[4::4], keys[7::4]))
        if 3072 in geo or geo.get(1024) == 1:  # projected CRS
            raise DataError(f"{p.name}: projected GeoTIFF; reproject it to WGS84 latitude/longitude")
    nodata = float(tags[42113]) if 42113 in tags else None
    area = keys is None or dict(zip(keys[4::4], keys[7::4])).get(1025, 1) == 1  # PixelIsArea
    lon0 = x + (0.5 * sx if area else 0) - i * sx
    lat0 = y - (0.5 * sy if area else 0) + j * sy
    _check_geographic(p.name, lon0, lat0, sx)
    return grid, lat0, lon0, sy, sx, nodata


def _check_geographic(name: str, x: float, y: float, cell: float) -> None:
    if not (abs(x) <= 180 and abs(y) <= 90 and cell < 0.1):
        raise DataError(f"{name}: coordinates look projected (metres); reproject to WGS84 latitude/longitude")


def load_dem(path: str | Path) -> ElevationModel:
    suffix = Path(path).suffix.lower()
    if suffix == ".asc":
        return load_esri_ascii(path)
    if suffix in (".tif", ".tiff"):
        return load_geotiff(path)
    raise DataError(f"unsupported DEM format '{suffix}' (expected .asc or .tif)")


# --------------------------------------------------------------------------- registration

@dataclass(frozen=True)
class DemRegistration:
    registration: GeoRegistration  # updated trajectory and points
    correction: Similarity  # applied on top of the previous ENU placement
    height_rms_m: float  # robust spread (1.4826 * MAD) of point-to-DEM height
    points_used: int
    shift_estimated: bool
    warnings: tuple[str, ...] = ()


def register_to_dem(reg: GeoRegistration, dem: ElevationModel, cfg: GeorefConfig | None = None
                    ) -> DemRegistration:
    cfg = cfg or GeorefConfig()
    P = reg.points
    ground = dem.height_at_enu(reg.frame, P[:, :2]) if len(P) else np.empty(0)
    ok = np.isfinite(ground)
    if ok.sum() < cfg.dem_min_points:
        raise DataError(f"DEM: only {int(ok.sum())} model points fall on the elevation model")
    P = P[ok]
    center = P.mean(0)
    relief = float(np.nanpercentile(ground[ok], 95) - np.nanpercentile(ground[ok], 5))
    estimate_shift = relief >= cfg.dem_min_relief_m
    sig_tilt = np.radians(max(float(np.hypot(*reg.rotation_std_deg[:2])), 0.05))
    sig_shift = max(reg.gps_rms_m, 0.5)

    def correction(x) -> Similarity:
        R = Rotation.from_rotvec([x[1], x[2], 0.0]).as_matrix()
        t = center - R @ center + [x[3], x[4], x[0]]
        return Similarity(1.0, R, t)

    def residuals(x):
        Q = correction(x).apply_points(P)
        h = dem.height_at_enu(reg.frame, Q[:, :2])
        h = np.where(np.isfinite(h), h, Q[:, 2])  # points pushed off the DEM contribute nothing
        return np.concatenate([(Q[:, 2] - h) / cfg.dem_sigma_m,
                               np.array([x[1], x[2]]) / sig_tilt,
                               np.array([x[3], x[4]]) / sig_shift])

    x0 = np.array([float(np.median(ground[ok] - P[:, 2])), 0, 0, 0, 0])
    free = [0, 1, 2, 3, 4] if estimate_shift else [0, 1, 2]

    def sub(xf):
        x = x0.copy()
        x[free] = xf
        return residuals(x)

    sol = least_squares(sub, x0[free], loss="soft_l1", f_scale=1.0, x_scale="jac")
    x = x0.copy()
    x[free] = sol.x
    corr = correction(x)

    Q = corr.apply_points(P)
    dh = Q[:, 2] - dem.height_at_enu(reg.frame, Q[:, :2])
    dh = dh[np.isfinite(dh)]
    inl = np.abs(dh) < 3 * cfg.dem_sigma_m
    # robust spread (scaled MAD): roofs and trees above bare earth should not inflate it
    rms = float(1.4826 * np.median(np.abs(dh - np.median(dh)))) if dh.size else float("inf")

    warnings: list[str] = []
    if not estimate_shift:
        warnings.append(f"terrain is too flat ({relief:.1f} m relief) to check horizontal position "
                        "against the elevation model; only height and tilt were adjusted")
    if abs(x[0]) > 3 * cfg.dem_sigma_m:
        warnings.append(f"model height moved {x[0]:+.1f} m to match the elevation model; GPS altitude and "
                        "DEM may use different vertical datums (ellipsoid vs mean sea level)")
    if inl.mean() < 0.5:
        warnings.append("most points sit well above or below the terrain; the DEM may be the wrong area "
                        "or a surface model rather than bare earth")

    full = reg.transform
    combined = Similarity(full.s, corr.R @ full.R, corr.R @ full.t + corr.t)
    new = replace(reg, transform=combined,
                  trajectory=replace(reg.trajectory, poses=tuple(corr.apply_pose(p) for p in reg.trajectory.poses)),
                  points=corr.apply_points(reg.points),
                  anchors=reg.anchors + ("dem",), warnings=reg.warnings + tuple(warnings))
    return DemRegistration(new, corr, rms, int(ok.sum()), estimate_shift, tuple(warnings))
