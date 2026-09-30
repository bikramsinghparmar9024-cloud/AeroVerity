"""Trust engine: one rule, three readable labels.

Continuous map: every surface vertex gets an effective 1-sigma in metres,

    effective = calibrated bootstrap sigma * inflation(cross-view agreement)

where agreement is evidence that widens the uncertainty (1 when all
neighbouring views agree, up to 1 + disagreement_inflation when none do, and
unscored_inflation where no view could check). The label is then decided by
that single number:

    MEASURED  effective <= measured_sigma_m
    INFERRED  effective <= inferred_sigma_m, or a surface exists but the
              bootstrap could not measure it
    UNKNOWN   anything worse, no real observations, or a moving object

Measurements are only answered between MEASURED vertices.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np
from scipy.ndimage import map_coordinates

from aerometric.config import TrustConfig
from aerometric.depth.crossview import CrossViewScore
from aerometric.depth.mono import AlignedDepth
from aerometric.pose.types import Trajectory
from aerometric.types import DataError


class TrustLabel(IntEnum):
    MEASURED = 0
    INFERRED = 1
    UNKNOWN = 2


@dataclass(frozen=True)
class TrustMap:
    labels: np.ndarray  # V uint8 TrustLabel
    effective_sigma_m: np.ndarray  # V, NaN where nothing could be estimated
    agreement: np.ndarray  # V, NaN where unscored
    reasons: np.ndarray  # V object: short reason for each non-MEASURED vertex ("" for MEASURED)
    evidence_is_mock: bool
    warnings: tuple[str, ...] = ()

    def fractions(self, vertices: np.ndarray | None = None, faces: np.ndarray | None = None) -> dict[str, float]:
        """Share of each label; area-weighted when a mesh is given (a face takes its worst vertex)."""
        if vertices is not None and faces is not None and len(faces):
            tri = vertices[faces]
            area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
            lab = self.labels[faces].max(1)
            total = area.sum() or 1.0
            return {TrustLabel(i).name.lower(): float(area[lab == i].sum() / total) for i in range(3)}
        n = max(len(self.labels), 1)
        return {TrustLabel(i).name.lower(): float(np.sum(self.labels == i) / n) for i in range(3)}


@dataclass(frozen=True)
class Measurement:
    allowed: bool
    value_m: float = float("nan")
    sigma_m: float = float("nan")
    reason: str = ""


# --------------------------------------------------------------------------- agreement in 3-D

def vertex_agreement(vertices: np.ndarray, traj: Trajectory, depths: Sequence[AlignedDepth | None],
                     scores: Sequence[CrossViewScore | None], cfg: TrustConfig | None = None) -> np.ndarray:
    """Mean cross-view agreement over the keyframes that actually see each vertex (NaN if none scored it)."""
    cfg = cfg or TrustConfig()
    K = traj.intrinsics
    total = np.zeros(len(vertices))
    count = np.zeros(len(vertices))
    for k, (d, sc) in enumerate(zip(depths, scores)):
        if d is None or sc is None or not d.good or not traj.valid[k]:
            continue
        p = traj.poses[k]
        Xc = vertices @ p.R_cw.T + p.t_cw
        z = Xc[:, 2]
        s = 1.0 / d.scale_to_full
        front = z > 1e-6
        u = np.where(front, (K.fx * Xc[:, 0] / np.where(front, z, 1) + K.cx) * s, -1)
        v = np.where(front, (K.fy * Xc[:, 1] / np.where(front, z, 1) + K.cy) * s, -1)
        h, w = d.depth.shape
        vis = front & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        if not vis.any():
            continue
        coords = [v[vis], u[vis]]
        dm = map_coordinates(d.depth.astype(float), coords, order=0, mode="nearest")
        ag = map_coordinates(sc.agreement.astype(float), coords, order=0, mode="nearest")
        ok = np.isfinite(dm) & (np.abs(dm - z[vis]) <= cfg.occlusion_tol * z[vis]) & np.isfinite(ag)
        idx = np.flatnonzero(vis)[ok]
        total[idx] += ag[ok]
        count[idx] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


# --------------------------------------------------------------------------- the rule

def assess(calibrated_sigma_m: np.ndarray, agreement: np.ndarray, weight: np.ndarray,
           dynamic: np.ndarray | None = None, evidence_is_mock: bool = False,
           cfg: TrustConfig | None = None) -> TrustMap:
    cfg = cfg or TrustConfig()
    n = len(calibrated_sigma_m)
    if not (len(agreement) == len(weight) == n) or (dynamic is not None and len(dynamic) != n):
        raise DataError("trust: all per-vertex inputs must have the same length")
    dynamic = np.zeros(n, bool) if dynamic is None else dynamic.astype(bool)

    infl = np.where(np.isfinite(agreement), 1 + cfg.disagreement_inflation * (1 - np.clip(agreement, 0, 1)),
                    cfg.unscored_inflation)
    eff = calibrated_sigma_m * infl

    labels = np.full(n, TrustLabel.UNKNOWN, np.uint8)
    reasons = np.full(n, "", dtype=object)
    measured = np.isfinite(eff) & (eff <= cfg.measured_sigma_m)
    inferred = (np.isfinite(eff) & (eff > cfg.measured_sigma_m) & (eff <= cfg.inferred_sigma_m)) | ~np.isfinite(eff)
    labels[measured] = TrustLabel.MEASURED
    labels[inferred] = TrustLabel.INFERRED
    reasons[inferred & np.isfinite(eff)] = "uncertainty above the measurement limit"
    reasons[inferred & ~np.isfinite(eff)] = "surface seen, but too few runs to estimate its uncertainty"
    reasons[np.isfinite(eff) & (eff > cfg.inferred_sigma_m)] = "uncertainty too large"

    unseen = ~(weight >= cfg.min_weight)
    labels[unseen] = TrustLabel.UNKNOWN
    reasons[unseen] = "not observed clearly enough"
    labels[dynamic] = TrustLabel.UNKNOWN
    reasons[dynamic] = "moving object"

    warnings: list[str] = []
    if evidence_is_mock:
        warnings.append("depth or segmentation came from a stand-in model; MEASURED areas rest on sparse points "
                        "interpolated between, so treat them with extra care")
    return TrustMap(labels, eff, agreement, reasons, evidence_is_mock, tuple(warnings))


# --------------------------------------------------------------------------- measurement gate

def measure_distance(trust: TrustMap, vertices: np.ndarray, a: int, b: int) -> Measurement:
    """Distance between two vertices, answered only when both are MEASURED."""
    for i in (a, b):
        if trust.labels[i] != TrustLabel.MEASURED:
            name = TrustLabel(int(trust.labels[i])).name.capitalize()
            return Measurement(False, reason=f"point {i} is {name} ({trust.reasons[i]}); "
                                             "measurement needs both points to be Measured")
    d = float(np.linalg.norm(vertices[a] - vertices[b]))
    sigma = float(np.hypot(trust.effective_sigma_m[a], trust.effective_sigma_m[b]))
    return Measurement(True, d, sigma)
