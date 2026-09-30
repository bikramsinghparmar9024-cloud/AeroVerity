"""Coverage check on the first sparse model, fed back to the quality gate.

The pre-flight gate only looks at images. Once poses and sparse points exist we
can check what was actually reconstructed:
  * ground coverage: the imaged footprint is rasterised on a ground grid and
    each cell is scored by point count, number of distinct observing keyframes
    and triangulation angle; weak cells are grouped into regions;
  * image spread: for each keyframe, how much of the frame holds tracked points.
Results come back as gate checks so the operator sees one combined report.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from aerometric.config import CoverageConfig
from aerometric.pose.classical import SparseModel
from aerometric.pose.tracks import max_ray_angle_deg
from aerometric.pose.types import Trajectory
from aerometric.quality.gate import CheckResult, GateReport, Status
from aerometric.types import DataError


@dataclass(frozen=True)
class GroundGrid:
    origin: tuple[float, float]  # ENU (east, north) of cell (0, 0)'s lower-left corner
    cell_m: float
    ground_z: float
    imaged: np.ndarray  # HxW bool, inside at least one camera footprint
    points: np.ndarray  # HxW int
    views: np.ndarray  # HxW int, distinct keyframes
    angle_deg: np.ndarray  # HxW float, median triangulation angle (nan where no points)
    weak: np.ndarray  # HxW bool, imaged but below a coverage threshold


@dataclass(frozen=True)
class WeakRegion:
    center_en: tuple[float, float]  # ENU metres
    area_m2: float
    cells: int


@dataclass(frozen=True)
class CoverageReport:
    grid: GroundGrid
    imaged_area_m2: float
    weak_fraction: float
    regions: tuple[WeakRegion, ...]  # largest first
    image_spread: np.ndarray  # per keyframe, share of image cells with points
    checks: tuple[CheckResult, ...]


# --------------------------------------------------------------------------- footprints

def ground_footprint(traj: Trajectory, k: int, z0: float) -> np.ndarray | None:
    """Image corners projected onto the plane z = z0 (Nx2 east/north), or None if not below."""
    K, p = traj.intrinsics, traj.poses[k]
    corners = np.array([[0, 0], [K.width, 0], [K.width, K.height], [0, K.height]], float)
    rays = np.column_stack([(corners - [K.cx, K.cy]) / [K.fx, K.fy], np.ones(4)]) @ p.R_cw
    C = p.center
    if np.any(rays[:, 2] >= -1e-6) or C[2] <= z0:
        return None  # a corner looks at or above the horizon: footprint is unbounded
    s = (z0 - C[2]) / rays[:, 2]
    return C[:2] + rays[:, :2] * s[:, None]


def _point_angles(traj: Trajectory, sparse: SparseModel) -> np.ndarray:
    return np.array([max_ray_angle_deg(traj.poses, tr[:, 0], X) for tr, X in zip(sparse.tracks, sparse.points)])


# --------------------------------------------------------------------------- analysis

def analyse_coverage(traj: Trajectory, sparse: SparseModel, keypoints_per_frame: list[np.ndarray] | None = None,
                     cfg: CoverageConfig | None = None) -> CoverageReport:
    cfg = cfg or CoverageConfig()
    if traj.world != "enu" or not traj.metric:
        raise DataError("coverage: trajectory must be metric ENU")
    if len(sparse.points) == 0:
        raise DataError("coverage: the sparse model has no points")

    z0 = float(np.median(sparse.points[:, 2]))
    feet = [f for k in range(len(traj)) if traj.valid[k] and (f := ground_footprint(traj, k, z0)) is not None]
    if not feet:
        raise DataError("coverage: no camera footprint intersects the ground")
    widths = [np.linalg.norm(f[1] - f[0]) for f in feet]
    cell = cfg.cell_m if cfg.cell_m > 0 else max(cfg.min_cell_m, float(np.median(widths)) / 40)

    allxy = np.concatenate(feet + [sparse.points[:, :2]])
    lo = np.floor(allxy.min(0) / cell) * cell
    shape = tuple((np.ceil((allxy.max(0) - lo) / cell).astype(int) + 1)[::-1])  # rows (N), cols (E)
    to_px = lambda xy: ((xy - lo) / cell).astype(np.int64)  # noqa: E731

    imaged = np.zeros(shape, np.uint8)
    for f in feet:
        cv2.fillPoly(imaged, [np.round((f - lo) / cell).astype(np.int32)], 1)
    # Only judge cells fully inside the footprint; rasterised edge cells are only partly imaged.
    imaged = cv2.erode(imaged, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT,
                       borderValue=0).astype(bool)

    pc = to_px(sparse.points[:, :2])
    rows, cols = pc[:, 1], pc[:, 0]
    flat = rows * shape[1] + cols
    points = np.bincount(flat, minlength=shape[0] * shape[1]).reshape(shape)

    cell_cam = np.unique(np.concatenate([np.column_stack([np.full(len(tr), c), tr[:, 0]])
                                         for c, tr in zip(flat, sparse.tracks)]), axis=0)
    views = np.bincount(cell_cam[:, 0], minlength=points.size).reshape(shape)

    ang = _point_angles(traj, sparse)
    angle = np.full(points.size, np.nan)
    order = np.argsort(flat)
    uniq, start = np.unique(flat[order], return_index=True)
    for u, a in zip(uniq, np.split(ang[order], start[1:])):
        angle[u] = np.median(a)
    angle = angle.reshape(shape)

    weak = imaged & ((points < cfg.min_points_per_cell) | (views < cfg.min_views_per_cell)
                     | ~(np.nan_to_num(angle, nan=0.0) >= cfg.min_angle_deg))
    grid = GroundGrid((float(lo[0]), float(lo[1])), cell, z0, imaged, points, views, angle, weak)

    regions = _regions(grid, cfg)
    imaged_n = int(imaged.sum())
    weak_fraction = float(weak.sum()) / imaged_n if imaged_n else 1.0
    spread = _image_spread(traj, keypoints_per_frame, sparse, cfg)
    checks = _checks(weak_fraction, regions, spread, traj, cfg)
    return CoverageReport(grid, imaged_n * cell * cell, weak_fraction, regions, spread, checks)


def _regions(grid: GroundGrid, cfg: CoverageConfig) -> tuple[WeakRegion, ...]:
    n, labels, stats, cent = cv2.connectedComponentsWithStats(grid.weak.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        cells = int(stats[i, cv2.CC_STAT_AREA])
        if cells < cfg.min_region_cells:
            continue
        cx, cy = cent[i]
        out.append(WeakRegion((grid.origin[0] + (cx + 0.5) * grid.cell_m, grid.origin[1] + (cy + 0.5) * grid.cell_m),
                              cells * grid.cell_m ** 2, cells))
    return tuple(sorted(out, key=lambda r: -r.cells))


def _image_spread(traj: Trajectory, keypoints: list[np.ndarray] | None, sparse: SparseModel,
                  cfg: CoverageConfig) -> np.ndarray:
    """Share of image grid cells that contain at least one observation of a sparse point."""
    gx, gy = cfg.image_grid
    K = traj.intrinsics
    occupied = [np.zeros((gy, gx), bool) for _ in range(len(traj))]
    if keypoints is None:
        return np.full(len(traj), np.nan)
    for tr in sparse.tracks:
        for c, k in tr:
            u, v = keypoints[c][k]
            occupied[c][min(gy - 1, max(0, int(v / K.height * gy))), min(gx - 1, max(0, int(u / K.width * gx)))] = True
    return np.array([o.mean() for o in occupied])


def _checks(weak_fraction: float, regions: tuple[WeakRegion, ...], spread: np.ndarray, traj: Trajectory,
            cfg: CoverageConfig) -> tuple[CheckResult, ...]:
    if weak_fraction >= cfg.fail_low_fraction:
        status: Status = "fail"
    elif weak_fraction >= cfg.warn_low_fraction:
        status = "warn"
    else:
        status = "pass"
    msg = f"{1 - weak_fraction:.0%} of the imaged ground is well covered."
    if regions:
        r = regions[0]
        msg += (f" Largest gap is about {r.area_m2:,.0f} m² around E {r.center_en[0]:.0f} m, "
                f"N {r.center_en[1]:.0f} m")
        msg += f" ({len(regions)} gaps in total)." if len(regions) > 1 else "."
    if status != "pass":
        msg += " Surfaces there will be marked Inferred or Unknown rather than Measured."
    checks = [CheckResult("Ground coverage", status, msg)]

    if np.all(np.isnan(spread)):
        return tuple(checks)
    thin = np.flatnonzero(spread < cfg.min_image_spread)
    if thin.size == 0:
        checks.append(CheckResult("Point spread", "pass",
                                  f"Tracked points are spread across every keyframe (median {np.nanmedian(spread):.0%} "
                                  "of each frame)."))
    else:
        frames = ", ".join(str(int(traj.frame_indices[i])) for i in thin[:5]) + ("…" if thin.size > 5 else "")
        checks.append(CheckResult("Point spread", "warn",
                                  f"{thin.size} keyframe(s) have points in only a small part of the frame "
                                  f"(frames {frames}); their poses rest on little evidence."))
    return tuple(checks)


# --------------------------------------------------------------------------- feedback to the gate

def with_coverage(gate: GateReport, coverage: CoverageReport) -> GateReport:
    """Merge coverage checks into the gate report and recompute its decision."""
    checks = tuple(c for c in gate.checks if c.name not in {k.name for k in coverage.checks}) + coverage.checks
    statuses = {c.status for c in checks}
    decision = "reject" if "fail" in statuses else "proceed_with_warnings" if "warn" in statuses else "proceed"
    return replace(gate, decision=decision, checks=checks)
