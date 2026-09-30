"""Multi-view tracks from pairwise matches, and linear triangulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from dronerecon.pose.types import CameraPose


@dataclass(frozen=True)
class PairMatches:
    i: int  # keyframe positions
    j: int
    idx_i: np.ndarray  # keypoint indices in image i
    idx_j: np.ndarray  # keypoint indices in image j


def build_tracks(n_keypoints: Sequence[int], pairs: Sequence[PairMatches]) -> list[np.ndarray]:
    """Union-find over (image, keypoint). Returns tracks as Kx2 int arrays [image, keypoint].

    Tracks that would contain two different keypoints of the same image are
    inconsistent (a bad match somewhere) and are discarded.
    """
    offset = np.concatenate([[0], np.cumsum(n_keypoints)])
    parent = np.arange(offset[-1])

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for p in pairs:
        for a, b in zip(offset[p.i] + p.idx_i, offset[p.j] + p.idx_j):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra

    groups: dict[int, list[int]] = {}
    matched = set()
    for p in pairs:
        matched.update((offset[p.i] + p.idx_i).tolist())
        matched.update((offset[p.j] + p.idx_j).tolist())
    for node in matched:
        groups.setdefault(find(node), []).append(node)

    image_of = np.searchsorted(offset, np.arange(offset[-1]), side="right") - 1
    tracks = []
    for nodes in groups.values():
        imgs = image_of[nodes]
        if len(nodes) < 2 or len(set(imgs.tolist())) != len(nodes):
            continue
        order = np.argsort(imgs)
        tracks.append(np.column_stack([imgs[order], np.asarray(nodes)[order] - offset[imgs[order]]]))
    return tracks


def triangulate(poses: Sequence[CameraPose], obs_norm: np.ndarray, cams: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation of one point from normalised observations."""
    rows = []
    for c, (x, y) in zip(cams, obs_norm):
        P = np.column_stack([poses[c].R_cw, poses[c].t_cw])
        rows += [x * P[2] - P[0], y * P[2] - P[1]]
    _, _, Vt = np.linalg.svd(np.asarray(rows))
    X = Vt[-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-12 else np.full(3, np.nan)


def max_ray_angle_deg(poses: Sequence[CameraPose], cams: np.ndarray, X: np.ndarray) -> float:
    rays = np.array([X - poses[c].center for c in cams])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    cos = np.clip(rays @ rays.T, -1, 1)
    return float(np.degrees(np.arccos(cos.min())))
