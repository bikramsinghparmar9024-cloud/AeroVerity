"""Render a synthetic drone flight over a procedurally generated residential area.

Outputs (in --out):
  flight.mp4               1280x720 video (MPEG-4)
  flight.srt               DJI-style per-frame telemetry, stamped at each frame's exact time
  camera_calibration.json  pinhole + OpenCV distortion model used to render the video
  ground_truth.csv         exact camera position and orientation per frame (for scoring)

The scene is a 2.5-D height field (ground, roofs, tree canopies, cars) with a
colour texture, lit by a sun with baked shadows. Frames are rendered by ray
marching the height field; walls get facade colours with windows. It is a clean
synthetic render, not photographic footage. GPS in the SRT carries a slow drift
like a consumer receiver; ground_truth.csv has the true positions.

    python tools/render_residential.py --out demo_data/residential --seconds 40 --fps 15
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# ----------------------------------------------------------------------------- world grid
RES = 0.25  # metres per cell
WX, WY = 600.0, 480.0  # world extent (east, north), centred on the origin
NX, NY = int(WX / RES), int(WY / RES)
GROUND_MSL = 212.0  # ground height above mean sea level
LAT0, LON0 = 28.6100, 77.2090  # a fictional neighbourhood; only the coordinates are real-looking
R_EARTH = 6_378_137.0


def ij(x, y):
    """World metres -> grid indices (column, row); row 0 is the north edge."""
    return (x + WX / 2) / RES, (WY / 2 - y) / RES


class World:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.h = np.zeros((NY, NX), np.float32)  # surface height (m)
        self.alb = np.zeros((NY, NX, 3), np.float32)  # top albedo, RGB 0..1
        self.facade = np.zeros((NY, NX, 3), np.float32)  # wall colour for building cells
        self.kind = np.zeros((NY, NX), np.uint8)  # 1 = building (walls get windows)
        yy, xx = np.mgrid[0:NY, 0:NX]
        self.X = (xx * RES - WX / 2 + RES / 2).astype(np.float32)
        self.Y = (WY / 2 - yy * RES - RES / 2).astype(np.float32)

    # ---------------------------------------------------------------- helpers
    def noise(self, scale_m: float, amp: float) -> np.ndarray:
        n = self.rng.standard_normal((NY, NX)).astype(np.float32)
        n = gaussian_filter(n, scale_m / RES)
        return n / (n.std() + 1e-6) * amp

    def rect(self, x0, y0, x1, y1):
        c0, r1 = ij(x0, y0)
        c1, r0 = ij(x1, y1)
        # clamp BOTH ends: a negative stop would silently mean "count from the end" and cover the map
        clamp = lambda v, n: int(min(n, max(0, v)))  # noqa: E731
        return (slice(clamp(np.floor(r0), NY), clamp(np.ceil(r1), NY)),
                slice(clamp(np.floor(c0), NX), clamp(np.ceil(c1), NX)))

    def paint(self, sl, rgb, var=0.06):
        n = self.rng.standard_normal(self.alb[sl].shape[:2] + (1,)).astype(np.float32) * var
        self.alb[sl] = np.clip(np.asarray(rgb, np.float32) * (1 + n), 0, 1)

    # ---------------------------------------------------------------- features
    def building(self, x0, y0, x1, y1, height, roof_rgb, wall_rgb, pitched=False):
        sl = self.rect(x0, y0, x1, y1)
        if pitched:  # ridge along the longer side
            X, Y = self.X[sl], self.Y[sl]
            if (x1 - x0) >= (y1 - y0):
                t = np.abs(Y - (y0 + y1) / 2) / ((y1 - y0) / 2)
            else:
                t = np.abs(X - (x0 + x1) / 2) / ((x1 - x0) / 2)
            ridge = 0.35 * min(x1 - x0, y1 - y0)
            self.h[sl] = height + ridge * (1 - np.clip(t, 0, 1))
            self.paint(sl, roof_rgb, 0.05)
            stripes = (np.floor((Y if (x1 - x0) >= (y1 - y0) else X) / 0.45) % 2)[..., None]
            self.alb[sl] *= (0.9 + 0.1 * stripes)  # roof tile rows
        else:
            self.h[sl] = height
            self.paint(sl, roof_rgb, 0.04)
            for _ in range(self.rng.integers(1, 4)):  # rooftop units
                cx, cy = self.rng.uniform(x0 + 2, x1 - 4), self.rng.uniform(y0 + 2, y1 - 4)
                s = self.rect(cx, cy, cx + self.rng.uniform(1.5, 3), cy + self.rng.uniform(1.5, 3))
                self.h[s] = height + self.rng.uniform(1, 2.5)
                self.paint(s, (0.55, 0.56, 0.58), 0.03)
        self.facade[sl] = wall_rgb
        self.kind[sl] = 1

    def tree(self, x, y):
        r, top = self.rng.uniform(2.2, 4.5), self.rng.uniform(6, 11)
        sl = self.rect(x - r, y - r, x + r, y + r)
        d2 = ((self.X[sl] - x) ** 2 + (self.Y[sl] - y) ** 2) / r ** 2
        canopy = np.where(d2 < 1, top - 2.5 * d2 + self.rng.normal(0, 0.15, d2.shape), 0)
        grow = canopy > self.h[sl]
        self.h[sl] = np.where(grow, canopy, self.h[sl])
        g = np.array([0.18, 0.36, 0.14]) * self.rng.uniform(0.8, 1.2)
        leaf = np.clip(g * (1 + self.rng.normal(0, 0.18, d2.shape + (1,))) * (1.1 - 0.3 * d2[..., None]), 0, 1)
        self.alb[sl] = np.where(grow[..., None], leaf, self.alb[sl])
        self.kind[sl] = np.where(grow, 0, self.kind[sl])

    def car(self, x, y, along_x):
        lx, ly = (4.4, 1.9) if along_x else (1.9, 4.4)
        sl = self.rect(x - lx / 2, y - ly / 2, x + lx / 2, y + ly / 2)
        self.h[sl] = 1.5
        col = self.rng.choice([(0.8, 0.8, 0.82), (0.1, 0.1, 0.12), (0.6, 0.1, 0.1), (0.15, 0.25, 0.55),
                               (0.55, 0.56, 0.58), (0.85, 0.75, 0.2)])
        self.paint(sl, col, 0.04)
        self.facade[sl] = np.asarray(col) * 0.8
        self.kind[sl] = 2

    # ---------------------------------------------------------------- blocks
    def houses(self, x0, y0, x1, y1):
        self.paint(self.rect(x0, y0, x1, y1), (0.30, 0.45, 0.20), 0.12)  # gardens
        pw, ph = 22.0, 30.0
        for px in np.arange(x0 + 1, x1 - pw + 1, pw + 2):
            for py in np.arange(y0 + 1, y1 - ph + 1, ph + 2):
                self.paint(self.rect(px, py, px + pw, py + 1.2), (0.62, 0.60, 0.56), 0.04)  # plot edge path
                w, d = self.rng.uniform(9, 13), self.rng.uniform(10, 14)
                hx, hy = px + (pw - w) / 2, py + self.rng.uniform(6, ph - d - 2)
                roof = self.rng.choice([(0.55, 0.22, 0.15), (0.35, 0.33, 0.33), (0.45, 0.30, 0.20), (0.25, 0.30, 0.40)])
                self.building(hx, hy, hx + w, hy + d, self.rng.uniform(5, 7.5), roof,
                              self.rng.choice([(0.85, 0.80, 0.70), (0.80, 0.72, 0.60), (0.9, 0.9, 0.88)]), pitched=True)
                self.paint(self.rect(hx + w / 2 - 1.5, py, hx + w / 2 + 1.5, hy), (0.55, 0.53, 0.50), 0.05)  # drive
                for _ in range(self.rng.integers(1, 4)):
                    self.tree(self.rng.uniform(px + 2, px + pw - 2), self.rng.uniform(py + 2, py + ph - 2))

    def apartments(self, x0, y0, x1, y1):
        self.paint(self.rect(x0, y0, x1, y1), (0.50, 0.50, 0.48), 0.05)
        n = self.rng.integers(1, 3)
        for k in range(n):
            bx0 = x0 + 6 + k * (x1 - x0 - 12) / n
            bx1 = bx0 + (x1 - x0 - 12) / n - 6
            self.building(bx0, y0 + 8, bx1, y1 - 14, self.rng.uniform(15, 30), (0.70, 0.70, 0.68),
                          self.rng.choice([(0.78, 0.74, 0.68), (0.70, 0.72, 0.75), (0.85, 0.70, 0.60)]))
        for x in np.arange(x0 + 4, x1 - 4, 2.6):  # parking row along the south side
            self.paint(self.rect(x, y0 + 1, x + 0.12, y0 + 6.5), (0.9, 0.9, 0.9), 0.02)
            if self.rng.random() < 0.6:
                self.car(x + 1.3, y0 + 3.7, along_x=False)

    def park(self, x0, y0, x1, y1):
        self.paint(self.rect(x0, y0, x1, y1), (0.28, 0.48, 0.18), 0.15)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        s = self.rect(x0, y0, x1, y1)
        d = np.hypot(self.X[s] - cx, self.Y[s] - cy)
        ring = np.abs(d - 22) < 1.6
        self.alb[s] = np.where(ring[..., None], np.array([0.72, 0.66, 0.52]) * (1 + self.rng.normal(0, .04, ring.shape + (1,))),
                               self.alb[s])  # walking loop
        pond = ((self.X[s] - cx - 5) / 12) ** 2 + ((self.Y[s] - cy + 3) / 8) ** 2 < 1
        self.alb[s] = np.where(pond[..., None], np.array([0.15, 0.30, 0.38]), self.alb[s])
        for _ in range(45):
            x, y = self.rng.uniform(x0 + 3, x1 - 3), self.rng.uniform(y0 + 3, y1 - 3)
            if abs(np.hypot(x - cx, y - cy) - 22) > 5 and ((x - cx - 5) / 14) ** 2 + ((y - cy + 3) / 10) ** 2 > 1:
                self.tree(x, y)

    def parking(self, x0, y0, x1, y1):
        self.paint(self.rect(x0, y0, x1, y1), (0.28, 0.28, 0.29), 0.06)
        for row, y in enumerate(np.arange(y0 + 4, y1 - 8, 14)):
            for x in np.arange(x0 + 3, x1 - 3, 2.7):
                self.paint(self.rect(x, y, x + 0.12, y + 5.5), (0.92, 0.92, 0.9), 0.02)
                if self.rng.random() < 0.7:
                    self.car(x + 1.35, y + 2.8, along_x=False)

    def community(self, x0, y0, x1, y1):
        s = self.rect(x0, y0, x1, y1)
        tiles = ((np.floor(self.X[s] / 1.5) + np.floor(self.Y[s] / 1.5)) % 2)[..., None]
        self.alb[s] = np.array([0.72, 0.66, 0.58]) * (0.92 + 0.08 * tiles) * (1 + self.rng.normal(0, .03, tiles.shape))
        self.building(x0 + 6, y0 + 6, x0 + 32, y0 + 30, 8, (0.62, 0.30, 0.22), (0.92, 0.88, 0.80))  # hall
        self.paint(self.rect(x1 - 30, y1 - 22, x1 - 8, y1 - 8), (0.25, 0.60, 0.80), 0.02)  # pool
        for k in range(8):
            self.tree(x1 - 34 + 3.5 * k, y0 + 8)

    # ---------------------------------------------------------------- city
    def build(self):
        self.paint((slice(None), slice(None)), (0.30, 0.46, 0.20), 0.1)
        spacing, road = 80.0, 10.0
        kinds = ["houses", "houses", "houses", "apartments", "apartments", "park", "parking", "community"]
        for bx in np.arange(-WX / 2, WX / 2, spacing):
            for by in np.arange(-WY / 2, WY / 2, spacing):
                x0, y0, x1, y1 = bx + road / 2 + 2, by + road / 2 + 2, bx + spacing - road / 2 - 2, by + spacing - road / 2 - 2
                getattr(self, str(self.rng.choice(kinds)))(x0, y0, x1, y1)
        for c in np.arange(-WX / 2, WX / 2 + 1, spacing):  # roads, sidewalks, markings
            self._road(c, vertical=True, width=road)
        for c in np.arange(-WY / 2, WY / 2 + 1, spacing):
            self._road(c, vertical=False, width=road)
        for cx in np.arange(-WX / 2, WX / 2 + 1, spacing):  # zebra crossings at intersections
            for cy in np.arange(-WY / 2, WY / 2 + 1, spacing):
                for k in range(-4, 5):
                    self.paint(self.rect(cx - road / 2 - 3.5, cy + k * 1.0, cx - road / 2 - 1.0, cy + k * 1.0 + 0.5),
                               (0.92, 0.92, 0.9), 0.02)
                    self.paint(self.rect(cx + k * 1.0, cy + road / 2 + 1.0, cx + k * 1.0 + 0.5, cy + road / 2 + 3.5),
                               (0.92, 0.92, 0.9), 0.02)
        self.alb *= (1 + self.noise(0.6, 0.06)[..., None])  # fine texture everywhere helps matching
        self.alb = np.clip(self.alb, 0, 1)

    def _road(self, c, vertical, width):
        half = width / 2
        if vertical:
            s = self.rect(c - half - 2, -WY / 2, c + half + 2, WY / 2)
            self.paint(s, (0.62, 0.61, 0.58), 0.05)
            s = self.rect(c - half, -WY / 2, c + half, WY / 2)
        else:
            s = self.rect(-WX / 2, c - half - 2, WX / 2, c + half + 2)
            self.paint(s, (0.62, 0.61, 0.58), 0.05)
            s = self.rect(-WX / 2, c - half, WX / 2, c + half)
        self.h[s] = 0
        self.kind[s] = 0
        self.paint(s, (0.20, 0.20, 0.22), 0.07)
        coord = self.Y[s] if vertical else self.X[s]
        dash = ((coord % 6) < 3)
        centre = np.abs((self.X[s] if vertical else self.Y[s]) - c) < 0.12
        self.alb[s] = np.where((dash & centre)[..., None], np.array([0.95, 0.9, 0.5]), self.alb[s])

    # ---------------------------------------------------------------- lighting
    def bake_light(self, sun_az_deg=140.0, sun_el_deg=48.0):
        az, el = np.radians(sun_az_deg), np.radians(sun_el_deg)
        self.sun = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], np.float32)
        gy, gx = np.gradient(self.h, RES)
        n = np.dstack([-gx, gy, np.ones_like(self.h)])  # rows go south, so +row is -north
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        lambert = np.clip(n @ self.sun, 0, 1)
        lit = np.ones_like(self.h)
        dx, dy = self.sun[0] / np.hypot(self.sun[0], self.sun[1]), self.sun[1] / np.hypot(self.sun[0], self.sun[1])
        rise = np.tan(el)
        yy, xx = np.mgrid[0:NY, 0:NX].astype(np.float32)
        for dist in np.arange(0.5, 45, 0.5):  # march toward the sun
            hh = map_coordinates(self.h, [yy - dy * dist / RES, xx + dx * dist / RES], order=0, mode="nearest")
            lit = np.minimum(lit, np.where(hh > self.h + dist * rise, 0.0, 1.0))
        lit = gaussian_filter(lit, 1.0)  # soft shadow edges
        self.top = np.clip(self.alb * (0.38 + 0.62 * (lambert * lit))[..., None], 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------- camera and flight
@dataclass(frozen=True)
class Camera:
    w: int = 1280
    h: int = 720
    fx: float = 905.0  # ~70 degree horizontal field of view
    fy: float = 905.0
    cx: float = 641.3
    cy: float = 358.7
    k1: float = -0.085
    k2: float = 0.021
    p1: float = 0.0004
    p2: float = -0.0003
    k3: float = 0.0


def pixel_rays(cam: Camera) -> np.ndarray:
    """Unit camera-frame ray per pixel, undistorting each pixel (the image is formed with distortion)."""
    u, v = np.meshgrid(np.arange(cam.w, dtype=np.float64), np.arange(cam.h, dtype=np.float64))
    pts = np.stack([u.ravel(), v.ravel()], 1)[:, None, :]
    K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])
    und = cv2.undistortPoints(pts, K, np.array([cam.k1, cam.k2, cam.p1, cam.p2, cam.k3]),
                              criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-9))[:, 0, :]
    rays = np.column_stack([und, np.ones(len(und))])
    return (rays / np.linalg.norm(rays, axis=1, keepdims=True)).astype(np.float32)


def camera_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """R_cw for compass yaw, gimbal pitch (-90 = nadir), roll; camera x right, y down, z forward."""
    h, p = np.radians(yaw_deg), np.radians(pitch_deg)
    fwd = np.array([np.sin(h), np.cos(h), 0.0])
    right = np.array([np.cos(h), -np.sin(h), 0.0])
    z = np.cos(p) * fwd + np.sin(p) * np.array([0.0, 0.0, 1.0])
    y = np.cross(z, right)
    R = np.array([right, y, z])
    r = np.radians(roll_deg)
    Rz = np.array([[np.cos(r), np.sin(r), 0], [-np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ R


def flight(t: np.ndarray, duration: float):
    """Smooth S-shaped pass: position (E, N, height above ground), heading, gimbal pitch, roll."""
    s = t / duration
    x = -150 + 300 * s
    y = 45 * np.sin(2 * np.pi * s * 0.9) - 10
    z = 70 + 12 * np.sin(2 * np.pi * s * 0.7 + 0.5)
    dx, dy = np.gradient(x, t), np.gradient(y, t)
    heading = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    pitch = -68 + 8 * np.sin(2 * np.pi * s * 1.3)
    roll = 0.4 * np.sin(2 * np.pi * s * 3.1)
    speed = np.hypot(dx, dy)
    vz = np.gradient(z, t)
    return np.column_stack([x, y, z]), heading, pitch, roll, speed, vz


# ----------------------------------------------------------------------------- renderer
def render(world: World, rays_cam: np.ndarray, R_cw: np.ndarray, C: np.ndarray, cam: Camera,
           step: float = 1.0, hmax: float = 34.0) -> np.ndarray:
    d = (rays_cam @ R_cw).astype(np.float32)  # world directions (rows of R_cw are camera axes)
    dz = np.minimum(d[:, 2], -1e-3)
    t0 = np.maximum((C[2] - hmax) / -dz, 0).astype(np.float32)
    t1 = (C[2] / -dz).astype(np.float32)
    n_steps = int(np.ceil((t1 - t0).max() / step))
    dt = (t1 - t0) / n_steps
    hit_t = t1.copy()
    prev_t = t0.copy()
    alive = np.ones(len(d), bool)
    hmap = world.h
    C32 = C.astype(np.float32)

    def height_nearest(t, idx):  # fast coarse march: nearest cell, plain integer indexing
        p = C32 + d[idx] * t[:, None]
        col = np.clip(((p[:, 0] + WX / 2) / RES).astype(np.int32), 0, NX - 1)
        row = np.clip(((WY / 2 - p[:, 1]) / RES).astype(np.int32), 0, NY - 1)
        return hmap[row, col], p[:, 2]

    def height_at(t, idx):  # precise: bilinear
        p = C + d[idx] * t[:, None]
        col, row = ij(p[:, 0], p[:, 1])
        return map_coordinates(hmap, [row - 0.5, col - 0.5], order=1, mode="nearest"), p[:, 2]

    for k in range(1, n_steps + 1):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        t = t0[idx] + dt[idx] * k
        hh, z = height_nearest(t, idx)
        hit = z <= hh
        hit_t[idx[hit]] = t[hit]
        prev_t[idx[hit]] = t[hit] - dt[idx[hit]]
        alive[idx[hit]] = False
    lo, hi = prev_t.astype(np.float64), hit_t.astype(np.float64)  # bisection refinement of the crossing
    all_idx = np.arange(len(d))
    for _ in range(8):
        mid = (lo + hi) / 2
        hh, z = height_at(mid, all_idx)
        below = z <= hh
        hi = np.where(below, mid, hi)
        lo = np.where(below, lo, mid)
    P = C + d * hi[:, None]
    col, row = ij(P[:, 0], P[:, 1])
    rc = [row - 0.5, col - 0.5]
    hh = map_coordinates(hmap, rc, order=1, mode="nearest")
    top = np.stack([map_coordinates(world.top[..., c], rc, order=1, mode="nearest") for c in range(3)], 1)

    # walls: the ray met a height step below the local top surface
    wall = (hh - P[:, 2]) > 0.35
    if wall.any():
        rw = [r[wall] for r in rc]
        fac = np.stack([map_coordinates(world.facade[..., c], rw, order=0, mode="nearest") for c in range(3)], 1)
        kind = map_coordinates(world.kind, rw, order=0, mode="nearest")
        gy = map_coordinates(np.gradient(hmap, axis=0), rw, order=1, mode="nearest")
        gx = map_coordinates(np.gradient(hmap, axis=1), rw, order=1, mode="nearest")
        nrm = np.stack([-gx, gy], 1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6
        sun_h = world.sun[:2] / np.linalg.norm(world.sun[:2])
        shade = 0.45 + 0.55 * np.clip(nrm @ sun_h, 0, 1)
        z, along = P[wall, 2], P[wall, 0] + P[wall, 1]
        window = (kind == 1) & ((z % 3.0) > 1.0) & ((z % 3.0) < 2.2) & ((along % 3.2) > 0.7) & ((along % 3.2) < 2.3)
        col_w = fac * shade[:, None]
        col_w[window] = np.array([0.18, 0.24, 0.30]) * (0.8 + 0.4 * shade[window, None])
        top[wall] = col_w
    rng = np.random.default_rng(int(C[0] * 1000) & 0xFFFF)
    img = top * (1 + rng.normal(0, 0.012, top.shape))  # sensor noise
    haze = 1 - np.exp(-hi / 900.0)
    img = img * (1 - haze[:, None]) + np.array([0.70, 0.76, 0.82]) * haze[:, None]
    img = (np.clip(img, 0, 1) ** (1 / 1.1) * 255).astype(np.uint8).reshape(cam.h, cam.w, 3)
    return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (0, 0), 0.6)  # lens softness


# ----------------------------------------------------------------------------- outputs
def enu_to_lla(e, n, u):
    lat = LAT0 + np.degrees(n / R_EARTH)
    lon = LON0 + np.degrees(e / (R_EARTH * np.cos(np.radians(LAT0))))
    return lat, lon, u + GROUND_MSL


def srt_time(s: float) -> str:
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def gps_drift(n: int, fps: float, sigma_m: float, rng) -> np.ndarray:
    k = max(1, int(fps * 2))  # ~2 s correlation, like a consumer receiver
    w = rng.standard_normal((n + k, 3))
    out = np.stack([np.convolve(w[:, i], np.ones(k) / np.sqrt(k), "valid")[:n] for i in range(3)], 1)
    return out * np.array([sigma_m, sigma_m, 1.5 * sigma_m])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="demo_data/residential")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--frames", type=int, default=0, help="render only this many frames (timing test)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cam = Camera()
    t_build = time.time()
    world = World(args.seed)
    world.build()
    world.bake_light()
    print(f"world built in {time.time() - t_build:.0f} s", flush=True)

    n = int(round(args.seconds * args.fps))
    t = np.arange(n) / args.fps
    pos, heading, pitch, roll, speed, vz = flight(t, args.seconds)
    rng = np.random.default_rng(args.seed + 1)
    gps = pos + gps_drift(n, args.fps, 0.45, rng)
    rays = pixel_rays(cam)

    n_render = args.frames or n
    writer = cv2.VideoWriter(str(out / "flight.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (cam.w, cam.h))
    start = datetime(2026, 9, 30, 10, 15, 0)
    srt_blocks, truth_rows = [], []
    t_render = time.time()
    for i in range(n_render):
        R = camera_rotation(heading[i], pitch[i], roll[i])
        writer.write(render(world, rays, R, pos[i], cam))
        lat, lon, alt = enu_to_lla(*gps[i])
        tl, tn, ta = enu_to_lla(*pos[i])
        stamp = start + timedelta(seconds=float(t[i]))
        srt_blocks.append(
            f"{i + 1}\n{srt_time(t[i])} --> {srt_time(t[i] + 1 / args.fps)}\n"
            f"<font size=\"28\">FrameCnt: {i + 1}, DiffTime: {int(round(1000 / args.fps))}ms\n"
            f"{stamp.strftime('%Y-%m-%d %H:%M:%S')}.{stamp.microsecond // 1000:03d}\n"
            f"[iso: 100] [shutter: 1/1000.0] [fnum: 2.8] [ev: 0] [focal_len: 24.00] "
            f"[latitude: {float(lat):.7f}] [longitude: {float(lon):.7f}] "
            f"[rel_alt: {gps[i, 2]:.3f} abs_alt: {float(alt):.3f}] "
            f"[h_speed: {speed[i]:.2f} v_speed: {vz[i]:.2f}] "
            f"[gb_yaw: {heading[i]:.1f} gb_pitch: {pitch[i]:.1f} gb_roll: {roll[i]:.1f}] </font>\n")
        truth_rows.append([i, f"{t[i]:.4f}", f"{float(tl):.8f}", f"{float(tn):.8f}", f"{float(ta):.3f}",
                           f"{pos[i, 0]:.3f}", f"{pos[i, 1]:.3f}", f"{pos[i, 2]:.3f}",
                           f"{heading[i]:.3f}", f"{pitch[i]:.3f}", f"{roll[i]:.3f}"])
        if i % 20 == 0 or i == n_render - 1:
            el = time.time() - t_render
            print(f"frame {i + 1}/{n_render}  ({el / (i + 1):.1f} s/frame, ~{el / (i + 1) * (n_render - i - 1) / 60:.0f} min left)",
                  flush=True)
    writer.release()

    (out / "flight.srt").write_text("\n".join(srt_blocks), encoding="utf-8")
    with (out / "ground_truth.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "true_lat", "true_lon", "true_alt_msl_m", "east_m", "north_m",
                    "height_above_ground_m", "heading_deg", "gimbal_pitch_deg", "gimbal_roll_deg"])
        w.writerows(truth_rows)
    calib = {
        "camera_model": "OPENCV",
        "image_width": cam.w, "image_height": cam.h,
        "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy,
        "camera_matrix": [[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]],
        "dist_coeffs": [cam.k1, cam.k2, cam.p1, cam.p2, cam.k3],
        "dist_model": "OpenCV (k1, k2, p1, p2, k3)",
        "horizontal_fov_deg": round(float(np.degrees(2 * np.arctan(cam.w / (2 * cam.fx)))), 2),
        "note": "Exact parameters used to render flight.mp4 (synthetic).",
    }
    (out / "camera_calibration.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
    print(f"done: {n_render} frames in {(time.time() - t_render) / 60:.1f} min -> {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
