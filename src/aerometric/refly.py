"""Re-fly advisor: where the model is weak, why, and where to point the camera next time.

Advice only; nothing here changes the model. Weak surfaces are the UNKNOWN and
INFERRED vertices from the trust engine (moving objects excluded; flying again
does not fix a parked car) plus ground gaps from the coverage check. They are
grouped into regions; for each region we look at how the existing keyframes saw
it, name the main reason, and propose two viewpoints facing the surface with
enough angle between them to triangulate it, kept above a minimum height.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from aerometric.config import ReflyConfig
from aerometric.geo import LocalFrame
from aerometric.pose.types import Trajectory
from aerometric.quality.coverage import WeakRegion
from aerometric.trust import TrustLabel, TrustMap


@dataclass(frozen=True)
class Viewpoint:
    position_enu: np.ndarray  # 3
    look_at_enu: np.ndarray  # 3
    heading_deg: float  # compass bearing the camera faces, 0 = north, clockwise
    gimbal_pitch_deg: float  # 0 = level, -90 = straight down
    lat: float = float("nan")
    lon: float = float("nan")
    alt_m: float = float("nan")


@dataclass(frozen=True)
class ReflyAdvice:
    center_enu: np.ndarray
    area_m2: float
    share_unknown: float  # of the region's vertices
    reason: str  # "never in view" | "seen by too few frames" | "seen from one direction only" | "grazing angle" | "coverage gap"
    existing_views: int
    viewpoints: tuple[Viewpoint, ...]
    message: str


# --------------------------------------------------------------------------- helpers

def _heading_pitch(look: np.ndarray) -> tuple[float, float]:
    heading = (np.degrees(np.arctan2(look[0], look[1])) + 360) % 360
    pitch = np.degrees(np.arctan2(look[2], np.hypot(look[0], look[1])))
    return float(heading), float(pitch)


def _compass(heading: float) -> str:
    names = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
    return names[int(((heading + 22.5) % 360) // 45)]


def _rotate(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    t = np.radians(deg)
    return v * np.cos(t) + np.cross(a, v) * np.sin(t) + a * (a @ v) * (1 - np.cos(t))


def _existing_views(traj: Trajectory, target: np.ndarray) -> np.ndarray:
    """Unit rays from target to every valid camera whose image contains it."""
    K = traj.intrinsics
    rays = []
    for p, ok in zip(traj.poses, traj.valid):
        if not ok:
            continue
        Xc = p.R_cw @ target + p.t_cw
        if Xc[2] <= 0:
            continue
        u, v = K.fx * Xc[0] / Xc[2] + K.cx, K.fy * Xc[1] / Xc[2] + K.cy
        if 0 <= u < K.width and 0 <= v < K.height:
            r = p.center - target
            rays.append(r / np.linalg.norm(r))
    return np.array(rays).reshape(-1, 3)


def _diagnose(rays: np.ndarray, normal: np.ndarray, cfg: ReflyConfig) -> str:
    if len(rays) == 0:
        return "never in view"
    if len(rays) < cfg.min_views:
        return "seen by too few frames"
    cosang = np.clip(rays @ rays.T, -1, 1)
    if np.degrees(np.arccos(cosang.min())) < cfg.min_angle_deg:
        return "seen from one direction only"
    if np.degrees(np.arccos(np.clip(rays @ normal, -1, 1))).min() > cfg.grazing_deg:
        return "grazing angle"
    return "seen by too few frames"  # enough views on paper, but the evidence still did not hold up


def _propose(center: np.ndarray, normal: np.ndarray, rays: np.ndarray, ground_z: float, standoff: float,
             frame: LocalFrame | None, cfg: ReflyConfig) -> tuple[Viewpoint, ...]:
    n = normal / (np.linalg.norm(normal) or 1.0)
    if abs(n[2]) < 0.7:
        # steep surface (wall, facade): spread the two views sideways so height limits cannot merge them
        axis = np.array([0.0, 0.0, 1.0])
    else:
        # roof or ground: spread across the direction the existing views came from, adding new parallax
        seen = rays.mean(0) if len(rays) else np.array([1.0, 0.0, 0.0])
        seen_h = np.array([seen[0], seen[1], 0.0])
        axis = seen_h if np.linalg.norm(seen_h) > 1e-3 else np.array([1.0, 0.0, 0.0])
        axis = np.cross(np.cross(n, axis), n)  # horizontal axis along the existing viewing direction
    views = []
    for sign in (-1, 1):
        d = _rotate(n, axis, sign * cfg.new_view_angle_deg / 2)
        pos = center + standoff * d
        floor = ground_z + cfg.min_altitude_agl_m
        if pos[2] < floor:  # lift toward the target's vertical until above the minimum height
            horiz = np.hypot(*(pos - center)[:2])
            pos = np.array([pos[0], pos[1], floor])
            if horiz < 1e-6:
                pos[2] = max(floor, center[2] + standoff)
        look = (center - pos) / np.linalg.norm(center - pos)
        heading, pitch = _heading_pitch(look)
        lat = lon = alt = float("nan")
        if frame is not None:
            la, lo, al = frame.to_lla(pos[None])
            lat, lon, alt = float(la[0]), float(lo[0]), float(al[0])
        views.append(Viewpoint(pos, center.copy(), heading, pitch, lat, lon, alt))
    return tuple(views)


def _message(adv_reason: str, area: float, share_unknown: float, views: tuple[Viewpoint, ...], center) -> str:
    what = "Unknown" if share_unknown >= 0.5 else "Inferred"
    v = views[0]
    rng = float(np.linalg.norm(v.position_enu - v.look_at_enu))
    side = _compass((v.heading_deg + 180) % 360)
    return (f"About {area:,.0f} m² around E {center[0]:.0f} m, N {center[1]:.0f} m is mostly {what} "
            f"({adv_reason}). Two shots from the {side}, about {rng:.0f} m away with the gimbal at "
            f"{v.gimbal_pitch_deg:.0f}°, should let it be measured.")


# --------------------------------------------------------------------------- main entry

def advise(vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray, trust: TrustMap, traj: Trajectory,
           dynamic: np.ndarray | None = None, gaps: Sequence[WeakRegion] = (), ground_z: float | None = None,
           frame: LocalFrame | None = None, cfg: ReflyConfig | None = None) -> list[ReflyAdvice]:
    cfg = cfg or ReflyConfig()
    weak = trust.labels != TrustLabel.MEASURED
    if dynamic is not None:
        weak &= ~dynamic
    gz = float(np.percentile(vertices[:, 2], 5)) if ground_z is None and len(vertices) else (ground_z or 0.0)
    cam_dist = np.linalg.norm(traj.centers[traj.valid][:, None] - vertices[None, ::max(1, len(vertices) // 500)],
                              axis=2) if len(vertices) else np.array([[50.0]])
    standoff = cfg.standoff_m if cfg.standoff_m > 0 else float(np.median(cam_dist.min(axis=0)))

    # vertex areas (a third of each adjacent face)
    tri = vertices[faces]
    fa = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    varea = np.bincount(faces.ravel(), np.repeat(fa / 3, 3), minlength=len(vertices))

    out: list[ReflyAdvice] = []
    idx = np.flatnonzero(weak)
    if idx.size:
        pairs = cKDTree(vertices[idx]).query_pairs(cfg.cluster_m, output_type="ndarray")
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(idx.size, idx.size))
        _, comp = connected_components(g, directed=False)
        for c in np.unique(comp):
            members = idx[comp == c]
            area = float(varea[members].sum())
            if area < cfg.min_area_m2:
                continue
            w = varea[members][:, None]
            center = (vertices[members] * w).sum(0) / w.sum()
            normal = (normals[members] * w).sum(0)
            normal = normal / (np.linalg.norm(normal) or 1.0)
            share_u = float(np.mean(trust.labels[members] == TrustLabel.UNKNOWN))
            rays = _existing_views(traj, center)
            reason = _diagnose(rays, normal, cfg)
            views = _propose(center, normal, rays, gz, standoff, frame, cfg)
            out.append(ReflyAdvice(center, area, share_u, reason, len(rays), views,
                                   _message(reason, area, share_u, views, center)))

    for gap in gaps:
        if gap.area_m2 < cfg.min_area_m2:
            continue
        center = np.array([gap.center_en[0], gap.center_en[1], gz])
        rays = _existing_views(traj, center)
        views = _propose(center, np.array([0.0, 0.0, 1.0]), rays, gz, standoff, frame, cfg)
        out.append(ReflyAdvice(center, gap.area_m2, 1.0, "coverage gap", len(rays), views,
                               _message("coverage gap", gap.area_m2, 1.0, views, center)))

    out.sort(key=lambda a: -a.area_m2 * (1 + a.share_unknown))
    return out[: cfg.max_regions]
