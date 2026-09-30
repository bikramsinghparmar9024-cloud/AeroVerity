"""Change map between two registered flights.

Both surfaces are on flight A's raster grid (B is rasterised onto it after
registration). For each cell:

    dz     = height_B - height_A
    sigma  = sqrt(sigma_A^2 + sigma_B^2 + registration_rms^2)

A cell counts as changed only when |dz| exceeds `significance` x sigma AND
`min_change_m`, so noise and honest uncertainty are not reported as change.
Cells where either flight has no surface, or its trust is Unknown, are marked
"not comparable" rather than "no change". Small patches are dropped; each
remaining patch becomes a polygon with its area, volume and height change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np

from aerometric.config import ChangeConfig
from aerometric.geo import LocalFrame
from aerometric.outputs.rasters import SurfaceRasters, write_geotiff
from aerometric.outputs.tiles import raster_to_geojson
from aerometric.trust import TrustLabel
from aerometric.types import DataError


class ChangeClass(IntEnum):
    NO_CHANGE = 0
    RAISED = 1  # material added: new building, stockpile, vegetation growth
    LOWERED = 2  # material removed: demolition, excavation, collapse
    NOT_COMPARABLE = 3  # a flight has no surface there, or it is Unknown
    NO_DATA = 255


@dataclass(frozen=True)
class ChangeArea:
    kind: str  # "raised" | "lowered"
    area_m2: float
    volume_m3: float  # signed: positive = added
    mean_dz_m: float
    max_abs_dz_m: float
    center_en: tuple[float, float]


@dataclass(frozen=True)
class ChangeMap:
    dz: np.ndarray  # HxW float32, NaN where not comparable
    sigma: np.ndarray  # HxW float32
    classes: np.ndarray  # HxW uint8 ChangeClass
    areas: tuple[ChangeArea, ...]
    comparable_fraction: float  # of cells where either flight has a surface
    summary: dict


def _cell_sigma(r: SurfaceRasters, fallback: float) -> np.ndarray:
    s = r.sigma_m.astype(float).copy()
    missing = ~np.isfinite(s) & (r.trust == TrustLabel.INFERRED)
    s[missing] = fallback
    return s


def compute_change(a: SurfaceRasters, b: SurfaceRasters, registration_rms_m: float = 0.0,
                   cfg: ChangeConfig | None = None) -> ChangeMap:
    cfg = cfg or ChangeConfig()
    if a.dsm.shape != b.dsm.shape or a.grid != b.grid:
        raise DataError("change: both flights must be rasterised on the same grid (flight A's)")
    g = a.grid
    cell_area = g.cell_m ** 2
    has_a, has_b = np.isfinite(a.dsm), np.isfinite(b.dsm)
    unknown = (a.trust == TrustLabel.UNKNOWN) | (b.trust == TrustLabel.UNKNOWN)
    sa, sb = _cell_sigma(a, cfg.fallback_sigma_m), _cell_sigma(b, cfg.fallback_sigma_m)
    sigma = np.sqrt(sa ** 2 + sb ** 2 + registration_rms_m ** 2)
    comparable = has_a & has_b & ~unknown & np.isfinite(sigma)

    dz = np.where(comparable, b.dsm.astype(float) - a.dsm.astype(float), np.nan)
    with np.errstate(invalid="ignore"):
        big = comparable & (np.abs(dz) >= cfg.min_change_m) & (np.abs(dz) > cfg.significance * sigma)
    raised, lowered = big & (dz > 0), big & (dz < 0)

    classes = np.full(dz.shape, ChangeClass.NO_DATA, np.uint8)
    classes[has_a | has_b] = ChangeClass.NOT_COMPARABLE
    classes[comparable] = ChangeClass.NO_CHANGE

    areas: list[ChangeArea] = []
    min_cells = max(1, int(np.ceil(cfg.min_area_m2 / cell_area)))
    E, N = g.centres()
    for mask, kind, code in ((raised, "raised", ChangeClass.RAISED), (lowered, "lowered", ChangeClass.LOWERED)):
        clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        n, lab = cv2.connectedComponents(clean.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            m = lab == i
            if m.sum() < min_cells:
                continue
            classes[m] = code
            d = dz[m]
            areas.append(ChangeArea(kind, float(m.sum() * cell_area), float(d.sum() * cell_area), float(d.mean()),
                                    float(np.abs(d).max()), (float(E[m].mean()), float(N[m].mean()))))

    either = has_a | has_b
    summary = {
        "comparable_fraction": float(comparable.sum() / max(either.sum(), 1)),
        "raised_area_m2": float(sum(x.area_m2 for x in areas if x.kind == "raised")),
        "lowered_area_m2": float(sum(x.area_m2 for x in areas if x.kind == "lowered")),
        "volume_added_m3": float(sum(x.volume_m3 for x in areas if x.kind == "raised")),
        "volume_removed_m3": float(-sum(x.volume_m3 for x in areas if x.kind == "lowered")),
        "change_areas": len(areas),
        "median_sigma_m": float(np.nanmedian(sigma[comparable])) if comparable.any() else float("nan"),
    }
    areas.sort(key=lambda x: -abs(x.volume_m3))
    return ChangeMap(dz.astype(np.float32), sigma.astype(np.float32), classes, tuple(areas),
                     summary["comparable_fraction"], summary)


def describe(cm: ChangeMap) -> list[str]:
    """Plain-language lines for the report and the operator."""
    s = cm.summary
    lines = [f"{s['comparable_fraction']:.0%} of the surveyed ground could be compared between the two flights."]
    if not cm.areas:
        lines.append("No change larger than the measurement uncertainty was found.")
    for a in cm.areas[:5]:
        verb = "rose" if a.kind == "raised" else "dropped"
        lines.append(f"About {a.area_m2:,.0f} m² around E {a.center_en[0]:.0f} m, N {a.center_en[1]:.0f} m "
                     f"{verb} by up to {a.max_abs_dz_m:.1f} m ({abs(a.volume_m3):,.0f} m³).")
    return lines


def write_change(folder: str | Path, cm: ChangeMap, a: SurfaceRasters, frame: LocalFrame,
                 nodata: float = -9999.0) -> dict[str, Path]:
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    g = a.grid
    dz = np.where(np.isfinite(cm.dz), cm.dz, nodata).astype(np.float32)
    files = {
        "dz": write_geotiff(out / "change_dz.tif", dz, g, frame, nodata, "Height change B - A, metres"),
        "classes": write_geotiff(out / "change_class.tif", cm.classes, g, frame, 255,
                                 "0 no change, 1 raised, 2 lowered, 3 not comparable, 255 no data"),
    }
    names = {int(ChangeClass.RAISED): "raised", int(ChangeClass.LOWERED): "lowered"}
    gj = raster_to_geojson(cm.classes, g, frame, names, "change",
                           skip=(int(ChangeClass.NO_CHANGE), int(ChangeClass.NOT_COMPARABLE), 255))
    E, N = g.centres()
    for f in gj["features"]:  # attach volume and height figures from the matching change area
        ring = np.array(f["geometry"]["coordinates"][0])
        lat, lon = ring[:, 1].mean(), ring[:, 0].mean()
        c = frame.to_enu(np.array([lat]), np.array([lon]), np.array([0.0]))[0]
        near = min((x for x in cm.areas if x.kind == f["properties"]["change"]),
                   key=lambda x: np.hypot(x.center_en[0] - c[0], x.center_en[1] - c[1]), default=None)
        if near is not None:
            f["properties"].update({"volume_m3": round(near.volume_m3, 1), "mean_dz_m": round(near.mean_dz_m, 2),
                                    "max_abs_dz_m": round(near.max_abs_dz_m, 2)})
    p = out / "change_areas.geojson"
    p.write_text(json.dumps(gj), encoding="utf-8")
    files["areas"] = p
    return files
