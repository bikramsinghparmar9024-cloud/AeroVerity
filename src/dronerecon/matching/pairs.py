"""Image-pair selection: sequence neighbours, GPS proximity and retrieval (loop closure).

Retrieval uses VLAD over the local descriptors already extracted for matching,
with a codebook learned from this flight's own features. It needs no weights,
works air-gapped and is deterministic for a given seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from dronerecon.config import PairingConfig
from dronerecon.matching.features import Features
from dronerecon.types import DataError


@dataclass(frozen=True)
class ImagePair:
    i: int  # position in the keyframe list, i < j
    j: int
    reasons: frozenset[str]  # subset of {"sequence", "gps", "retrieval"}
    similarity: float = float("nan")  # retrieval similarity when available


# --------------------------------------------------------------------------- VLAD retrieval

class VladRetrieval:
    def __init__(self, clusters: int = 32, seed: int = 0) -> None:
        self.k, self.seed = clusters, seed
        self.centers: np.ndarray | None = None

    def fit(self, features: Sequence[Features], max_samples: int = 50_000) -> "VladRetrieval":
        desc = np.concatenate([f.descriptors for f in features if len(f)], axis=0)
        if len(desc) < self.k:
            raise DataError(f"retrieval: only {len(desc)} descriptors to build {self.k} clusters")
        rng = np.random.default_rng(self.seed)
        if len(desc) > max_samples:
            desc = desc[rng.choice(len(desc), max_samples, replace=False)]
        cv2.setRNGSeed(self.seed)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
        _, _, centers = cv2.kmeans(desc.astype(np.float32), self.k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        self.centers = centers
        return self

    def encode(self, f: Features) -> np.ndarray:
        if self.centers is None:
            raise DataError("retrieval: call fit() before encode()")
        k, d = self.centers.shape
        v = np.zeros((k, d), np.float32)
        if len(f):
            dist = ((f.descriptors[:, None, :] - self.centers[None]) ** 2).sum(-1)
            nearest = dist.argmin(1)
            np.add.at(v, nearest, f.descriptors - self.centers[nearest])
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12  # intra-normalisation
        v = v.ravel()
        v = np.sign(v) * np.sqrt(np.abs(v))  # power normalisation
        return v / (np.linalg.norm(v) + 1e-12)


def global_descriptors(features: Sequence[Features], cfg: PairingConfig | None = None,
                       seed: int = 0) -> np.ndarray:
    cfg = cfg or PairingConfig()
    vlad = VladRetrieval(cfg.vlad_clusters, seed).fit(features)
    return np.stack([vlad.encode(f) for f in features])


# --------------------------------------------------------------------------- pair selection

def select_pairs(n: int, cfg: PairingConfig | None = None, positions: np.ndarray | None = None,
                 descriptors: np.ndarray | None = None) -> list[ImagePair]:
    """Pairs among `n` keyframes (in flight order).

    positions: Nx3 ENU camera centres (smoothed GPS), optional.
    descriptors: NxD L2-normalised global descriptors, optional.
    """
    cfg = cfg or PairingConfig()
    if positions is not None and len(positions) != n:
        raise DataError(f"pairs: {len(positions)} positions for {n} keyframes")
    if descriptors is not None and len(descriptors) != n:
        raise DataError(f"pairs: {len(descriptors)} descriptors for {n} keyframes")

    reasons: dict[tuple[int, int], set[str]] = {}
    sims: dict[tuple[int, int], float] = {}

    def add(a: int, b: int, why: str) -> None:
        if a != b:
            reasons.setdefault((min(a, b), max(a, b)), set()).add(why)

    for i in range(n):
        for j in range(i + 1, min(n, i + 1 + cfg.sequence_window)):
            add(i, j, "sequence")

    dist = None
    if positions is not None:
        dist = np.linalg.norm(positions[:, None, :] - positions[None], axis=-1)
        for i in range(n):
            order = [j for j in np.argsort(dist[i]) if j != i and dist[i, j] <= cfg.gps_radius_m]
            for j in order[: cfg.gps_max_neighbors]:
                add(i, int(j), "gps")

    if descriptors is not None:
        sim = descriptors @ descriptors.T
        for i in range(n):
            far_in_time = [j for j in np.argsort(-sim[i])
                           if abs(j - i) > cfg.sequence_window and sim[i, j] >= cfg.retrieval_min_similarity]
            if dist is not None and cfg.retrieval_gps_gate_m > 0:
                far_in_time = [j for j in far_in_time if dist[i, j] <= cfg.retrieval_gps_gate_m]
            for j in far_in_time[: cfg.retrieval_top_k]:
                add(i, int(j), "retrieval")
                sims[(min(i, int(j)), max(i, int(j)))] = float(sim[i, j])

    return [ImagePair(i, j, frozenset(r), sims.get((i, j), float("nan")))
            for (i, j), r in sorted(reasons.items())]
