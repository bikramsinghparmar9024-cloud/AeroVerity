"""Vote per-keyframe segmentation labels into 3-D points.

Each point is projected into every keyframe that has a segmentation. If the
keyframe actually sees the point (its depth map agrees with the point's depth
at that pixel), the pixel's label is a vote, weighted by the segmenter's
confidence. Without depth maps there is no occlusion test: a hidden point would
take the label of whatever covers it, and the result says so.

Points whose winning votes are people, vehicles or animals are flagged as
dynamic so fusion can leave them out of the surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import map_coordinates

from dronerecon.config import SemanticConfig
from dronerecon.depth.mono import AlignedDepth
from dronerecon.pose.types import Trajectory
from dronerecon.segmentation import DYNAMIC, Label, SegmentationResult
from dronerecon.types import DataError

NO_LABEL = 255
N_LABELS = len(Label)


@dataclass(frozen=True)
class SemanticPoints:
    labels: np.ndarray  # N uint8, Label value or NO_LABEL
    confidence: np.ndarray  # N float, winning share of the vote (0 when unlabelled)
    votes: np.ndarray  # N int, keyframes that saw the point
    dynamic: np.ndarray  # N bool
    occlusion_checked: bool
    is_mock: bool  # any vote came from a placeholder segmenter
    warnings: tuple[str, ...] = ()

    def class_counts(self) -> dict[str, int]:
        out = {Label(i).name.lower(): int(c) for i, c in enumerate(np.bincount(
            self.labels[self.labels != NO_LABEL], minlength=N_LABELS)) if c}
        if (n := int((self.labels == NO_LABEL).sum())):
            out["unlabelled"] = n
        return out


def vote_labels(points: np.ndarray, traj: Trajectory, segs: Sequence[SegmentationResult | None],
                depths: Sequence[AlignedDepth | None] | None = None,
                cfg: SemanticConfig | None = None) -> SemanticPoints:
    cfg = cfg or SemanticConfig()
    if len(segs) != len(traj):
        raise DataError("semantic: need one segmentation (or None) per keyframe")
    if depths is not None and len(depths) != len(traj):
        raise DataError("semantic: need one depth map (or None) per keyframe")
    n = len(points)
    tally = np.zeros((n, N_LABELS))
    seen = np.zeros(n, int)
    K = traj.intrinsics
    used_mock = False

    for k, seg in enumerate(segs):
        if seg is None or not traj.valid[k]:
            continue
        p = traj.poses[k]
        Xc = points @ p.R_cw.T + p.t_cw
        z = Xc[:, 2]
        front = z > 1e-6
        u = np.where(front, K.fx * Xc[:, 0] / np.where(front, z, 1) + K.cx, -1)
        v = np.where(front, K.fy * Xc[:, 1] / np.where(front, z, 1) + K.cy, -1)
        H, W = seg.labels.shape
        vis = front & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        if depths is not None and depths[k] is not None and depths[k].good:
            d = depths[k]
            s = 1.0 / d.scale_to_full
            dm = np.full(n, np.nan)
            dm[vis] = map_coordinates(d.depth.astype(float), [v[vis] * s, u[vis] * s], order=0, mode="nearest")
            vis &= np.isfinite(dm) & (np.abs(dm - z) <= cfg.occlusion_tol * z)
        idx = np.flatnonzero(vis)
        if idx.size == 0:
            continue
        ui, vi = np.round(u[idx]).astype(int), np.round(v[idx]).astype(int)
        lab = seg.labels[vi, ui].astype(int)
        w = seg.confidence[vi, ui] if seg.confidence is not None else np.ones(idx.size)
        np.add.at(tally, (idx, lab), w)
        seen[idx] += 1
        used_mock |= seg.is_mock

    total = tally.sum(1)
    win = tally.argmax(1)
    labelled = (seen >= cfg.min_votes) & (total > 0)
    labels = np.where(labelled, win, NO_LABEL).astype(np.uint8)
    conf = np.where(labelled, tally.max(1) / np.where(total > 0, total, 1), 0.0)
    dyn_share = tally[:, [int(c) for c in DYNAMIC]].sum(1) / np.where(total > 0, total, 1)
    dynamic = labelled & (dyn_share >= cfg.dynamic_share)

    checked = depths is not None and any(d is not None and d.good for d in depths)
    warnings = []
    if not checked:
        warnings.append("no depth maps were available, so hidden points took the label of whatever covers them")
    if used_mock:
        warnings.append("labels come from the placeholder segmenter (vegetation and water only)")
    if (un := int((~labelled).sum())):
        warnings.append(f"{un} point(s) were not seen clearly by any keyframe and stay unlabelled")
    return SemanticPoints(labels, conf, seen, dynamic, checked, used_mock, tuple(warnings))
