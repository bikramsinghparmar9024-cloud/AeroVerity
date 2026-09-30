"""Local tangent-plane coordinates.

Equirectangular approximation around a fixed origin: sub-decimetre error
within a few kilometres, which covers a single-pass survey. Georeferencing
(Phase 8) replaces this with a proper projection for exported products.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True)
class LocalFrame:
    lat0: float
    lon0: float
    alt0: float

    def to_enu(self, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> np.ndarray:
        """Return Nx3 east/north/up metres."""
        c = np.cos(np.radians(self.lat0))
        e = np.radians(np.asarray(lon) - self.lon0) * EARTH_RADIUS_M * c
        n = np.radians(np.asarray(lat) - self.lat0) * EARTH_RADIUS_M
        return np.column_stack([e, n, np.asarray(alt, dtype=float) - self.alt0])

    def to_lla(self, enu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        enu = np.atleast_2d(enu)
        c = np.cos(np.radians(self.lat0))
        lat = self.lat0 + np.degrees(enu[:, 1] / EARTH_RADIUS_M)
        lon = self.lon0 + np.degrees(enu[:, 0] / (EARTH_RADIUS_M * c))
        return lat, lon, enu[:, 2] + self.alt0


# --------------------------------------------------------------------------- WGS84 ellipsoid (exact)

WGS84_A = 6_378_137.0
WGS84_E2 = 6.69437999014e-3


def lla_to_ecef(lat: np.ndarray, lon: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Geodetic latitude/longitude (degrees) and ellipsoidal height (m) -> Nx3 ECEF metres."""
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(la) ** 2)
    h = np.asarray(h, float)
    return np.column_stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                            (n * (1 - WGS84_E2) + h) * np.sin(la)])


def enu_basis(lat: float, lon: float) -> np.ndarray:
    """3x3 whose rows are the east, north and up unit vectors in ECEF at a geodetic position."""
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([[-np.sin(lo), np.cos(lo), 0.0],
                     [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                     [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])


def enu_to_ecef_matrix(lat: float, lon: float, h: float) -> np.ndarray:
    """4x4 transform from the exact WGS84 tangent-plane ENU frame at a point to ECEF."""
    M = np.eye(4)
    M[:3, :3] = enu_basis(lat, lon).T
    M[:3, 3] = lla_to_ecef(np.array([lat]), np.array([lon]), np.array([h]))[0]
    return M
