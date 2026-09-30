"""Feature extraction and matching.

Primary backend: ALIKED or DISK features + LightGlue (the `lightglue` package),
used automatically when it is installed. Fallback: OpenCV SIFT with mutual
nearest-neighbour + ratio test. The fallback is weaker on blur, low texture and
large viewpoint change; results carry `is_fallback` so reports can say so.

Keypoints are always returned in full-resolution pixel coordinates. Pixels
where `exclude` is True (moving objects, sky, water) never produce keypoints.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from aerometric.config import MatchingConfig
from aerometric.types import DataError


@dataclass(frozen=True)
class Features:
    keypoints: np.ndarray  # Nx2 float32, full-resolution pixels (x, y)
    descriptors: np.ndarray  # NxD float32
    scores: np.ndarray  # N float32
    image_size: tuple[int, int]  # (width, height)

    def __len__(self) -> int:
        return len(self.keypoints)


@dataclass(frozen=True)
class Matches:
    idx0: np.ndarray  # M int, indices into features of image 0
    idx1: np.ndarray  # M int, indices into features of image 1
    scores: np.ndarray  # M float32, higher is better

    def __len__(self) -> int:
        return len(self.idx0)

    def subset(self, keep: np.ndarray) -> "Matches":
        return Matches(self.idx0[keep], self.idx1[keep], self.scores[keep])


class Extractor(Protocol):
    name: str
    is_fallback: bool

    def extract(self, image: np.ndarray, exclude: np.ndarray | None = None) -> Features: ...


class Matcher(Protocol):
    name: str
    is_fallback: bool

    def match(self, f0: Features, f1: Features) -> Matches: ...


# --------------------------------------------------------------------------- helpers

def _prepare(image: np.ndarray, exclude: np.ndarray | None, max_dim: int
             ) -> tuple[np.ndarray, np.ndarray | None, float]:
    if image.ndim != 3:
        raise DataError("extract: expected a BGR colour image")
    h, w = image.shape[:2]
    if exclude is not None and exclude.shape != (h, w):
        raise DataError(f"extract: mask shape {exclude.shape} != image shape {(h, w)}")
    s = min(1.0, max_dim / max(h, w))
    size = (round(w * s), round(h * s))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if s < 1 else image
    mask = None
    if exclude is not None:
        mask = cv2.resize(exclude.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    return small, mask, s


def _drop_masked(kp: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None or len(kp) == 0:
        return np.ones(len(kp), bool)
    x = np.clip(kp[:, 0].round().astype(int), 0, mask.shape[1] - 1)
    y = np.clip(kp[:, 1].round().astype(int), 0, mask.shape[0] - 1)
    return ~mask[y, x]


# --------------------------------------------------------------------------- OpenCV fallback

class SiftExtractor:
    name = "sift"
    is_fallback = True

    def __init__(self, cfg: MatchingConfig) -> None:
        self.cfg = cfg
        self._sift = cv2.SIFT_create(nfeatures=cfg.max_keypoints)

    def extract(self, image: np.ndarray, exclude: np.ndarray | None = None) -> Features:
        small, mask, s = _prepare(image, exclude, self.cfg.max_dim)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        cv_mask = None if mask is None else (~mask).astype(np.uint8) * 255
        kps, desc = self._sift.detectAndCompute(grey, cv_mask)
        h, w = image.shape[:2]
        if not kps:
            return Features(np.empty((0, 2), np.float32), np.empty((0, 128), np.float32),
                            np.empty(0, np.float32), (w, h))
        # RootSIFT: more reliable nearest-neighbour matching than raw SIFT
        desc = desc / (np.abs(desc).sum(axis=1, keepdims=True) + 1e-7)
        desc = np.sqrt(desc).astype(np.float32)
        pts = np.array([k.pt for k in kps], np.float32) / s
        return Features(pts, desc, np.array([k.response for k in kps], np.float32), (w, h))


class MutualRatioMatcher:
    name = "mnn-ratio"
    is_fallback = True

    def __init__(self, cfg: MatchingConfig) -> None:
        self.cfg = cfg

    def match(self, f0: Features, f1: Features) -> Matches:
        empty = Matches(np.empty(0, int), np.empty(0, int), np.empty(0, np.float32))
        if len(f0) < 2 or len(f1) < 2:
            return empty
        bf = cv2.BFMatcher(cv2.NORM_L2)
        fwd = bf.knnMatch(f0.descriptors, f1.descriptors, k=2)
        bwd = {m.queryIdx: m.trainIdx for m in bf.match(f1.descriptors, f0.descriptors)}
        i0, i1, sc = [], [], []
        for pair in fwd:
            if len(pair) < 2:
                continue
            a, b = pair
            if a.distance < self.cfg.ratio * b.distance and bwd.get(a.trainIdx) == a.queryIdx:
                i0.append(a.queryIdx)
                i1.append(a.trainIdx)
                sc.append(1.0 - a.distance / max(b.distance, 1e-7))
        if not i0:
            return empty
        return Matches(np.array(i0), np.array(i1), np.array(sc, np.float32))


# --------------------------------------------------------------------------- LightGlue backend

def lightglue_available() -> bool:
    return importlib.util.find_spec("lightglue") is not None


class LightGlueBackend:
    """ALIKED/DISK + LightGlue. Acts as both extractor and matcher."""

    is_fallback = False

    def __init__(self, cfg: MatchingConfig) -> None:
        if not lightglue_available():
            raise DataError("the 'lightglue' package is not installed; install it or set "
                            "matching.backend to 'opencv'")
        if cfg.extractor not in ("aliked", "disk"):
            raise DataError(f"matching.extractor must be 'aliked' or 'disk', got '{cfg.extractor}'")
        import torch
        import lightglue

        self.cfg = cfg
        self.name = f"{cfg.extractor}+lightglue"
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        make = lightglue.ALIKED if cfg.extractor == "aliked" else lightglue.DISK
        self._extractor = make(max_num_keypoints=cfg.max_keypoints).eval().to(self._device)
        self._matcher = lightglue.LightGlue(features=cfg.extractor).eval().to(self._device)

    def extract(self, image: np.ndarray, exclude: np.ndarray | None = None) -> Features:
        small, mask, s = _prepare(image, exclude, self.cfg.max_dim)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        t = self._torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self._device)
        with self._torch.inference_mode():
            out = self._extractor.extract(t)
        kp = out["keypoints"][0].cpu().numpy()
        keep = _drop_masked(kp, mask)
        h, w = image.shape[:2]
        return Features((kp[keep] / s).astype(np.float32),
                        out["descriptors"][0].cpu().numpy()[keep].astype(np.float32),
                        out["keypoint_scores"][0].cpu().numpy()[keep].astype(np.float32), (w, h))

    def match(self, f0: Features, f1: Features) -> Matches:
        tt = self._torch

        def pack(f: Features) -> dict:
            return {"keypoints": tt.from_numpy(f.keypoints)[None].to(self._device),
                    "descriptors": tt.from_numpy(f.descriptors)[None].to(self._device),
                    "image_size": tt.tensor([f.image_size], dtype=tt.float32, device=self._device)}

        with tt.inference_mode():
            out = self._matcher({"image0": pack(f0), "image1": pack(f1)})
        m = out["matches"][0].cpu().numpy()
        sc = out["scores"][0].cpu().numpy().astype(np.float32)
        return Matches(m[:, 0], m[:, 1], sc)


# --------------------------------------------------------------------------- factory & verification

def make_matching(cfg: MatchingConfig | None = None) -> tuple[Extractor, Matcher]:
    cfg = cfg or MatchingConfig()
    if cfg.backend == "lightglue" or (cfg.backend == "auto" and lightglue_available()):
        lg = LightGlueBackend(cfg)
        return lg, lg
    if cfg.backend in ("opencv", "auto"):
        return SiftExtractor(cfg), MutualRatioMatcher(cfg)
    raise DataError(f"matching.backend must be 'auto', 'lightglue' or 'opencv', got '{cfg.backend}'")


@dataclass(frozen=True)
class VerifiedPair:
    matches: Matches  # epipolar inliers only
    F: np.ndarray | None  # fundamental matrix, None if verification failed
    raw_count: int

    @property
    def inlier_ratio(self) -> float:
        return len(self.matches) / self.raw_count if self.raw_count else 0.0


def verify_pair(f0: Features, f1: Features, m: Matches,
                cfg: MatchingConfig | None = None, seed: int = 0) -> VerifiedPair:
    """Keep matches consistent with a single fundamental matrix (MAGSAC++)."""
    cfg = cfg or MatchingConfig()
    none = VerifiedPair(m.subset(np.zeros(len(m), bool)), None, len(m))
    if len(m) < max(8, cfg.min_inliers):
        return none
    cv2.setRNGSeed(seed)
    F, inl = cv2.findFundamentalMat(f0.keypoints[m.idx0], f1.keypoints[m.idx1], cv2.USAC_MAGSAC,
                                    cfg.ransac_px, 0.999, 10000)
    if F is None or inl is None:
        return none
    keep = inl.ravel().astype(bool)
    if keep.sum() < cfg.min_inliers:
        return none
    return VerifiedPair(m.subset(keep), F[:3], len(m))
