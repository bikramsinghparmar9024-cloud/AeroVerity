"""Segmenter contract, label set and the placeholder backend.

Every backend returns per-pixel labels (and optionally a confidence) at the
input image's resolution. Downstream code only relies on:
  * `SegmentationResult.exclusion_mask()`  -> pixels matching must ignore
  * `SegmentationResult.labels`            -> classes voted into 3D later
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from dronerecon.config import SegmentationConfig
from dronerecon.types import DataError


class Label(IntEnum):
    OTHER = 0  # ground, bare soil, anything not covered below
    ROAD = 1
    BUILDING = 2
    VEGETATION = 3
    WATER = 4
    SKY = 5
    PERSON = 6
    VEHICLE = 7
    ANIMAL = 8


DYNAMIC = frozenset({Label.PERSON, Label.VEHICLE, Label.ANIMAL})
EXCLUDED_FROM_MATCHING = DYNAMIC | {Label.SKY, Label.WATER}


@dataclass(frozen=True)
class SegmentationResult:
    labels: np.ndarray  # HxW uint8, values from Label
    confidence: np.ndarray | None  # HxW float32 in [0, 1]
    backend: str
    is_mock: bool

    def __post_init__(self) -> None:
        if self.labels.ndim != 2 or self.labels.dtype != np.uint8:
            raise DataError("SegmentationResult.labels must be a 2-D uint8 array")
        if self.confidence is not None and self.confidence.shape != self.labels.shape:
            raise DataError("SegmentationResult.confidence must match labels shape")

    def mask_of(self, classes: frozenset[Label] | set[Label]) -> np.ndarray:
        return np.isin(self.labels, [int(c) for c in classes])

    def exclusion_mask(self, dilate_px: int = 0) -> np.ndarray:
        """True where features must not be extracted (moving objects, sky, water)."""
        mask = self.mask_of(EXCLUDED_FROM_MATCHING)
        if dilate_px > 0 and mask.any():
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
            mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
        return mask

    def class_fractions(self) -> dict[str, float]:
        counts = np.bincount(self.labels.ravel(), minlength=len(Label))
        return {Label(i).name.lower(): float(c) / self.labels.size for i, c in enumerate(counts) if c}


@runtime_checkable
class Segmenter(Protocol):
    name: str
    is_mock: bool

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Segment a BGR uint8 image; result has the same height and width."""
        ...


class MockSegmenter:
    """PLACEHOLDER, not a learned model.

    Colour/texture rules so the pipeline can run end to end: saturated green ->
    vegetation, smooth blue -> water. It never detects people, vehicles, animals,
    roads or buildings, so dynamic-object masking is effectively off while this
    backend is in use. Confidence is fixed low to make that visible downstream.
    """

    name = "mock"
    is_mock = True
    CONFIDENCE = 0.3

    def __init__(self, cfg: SegmentationConfig | None = None) -> None:
        self.cfg = cfg or SegmentationConfig()

    def segment(self, image: np.ndarray) -> SegmentationResult:
        if image.ndim != 3 or image.shape[2] != 3:
            raise DataError("segment: expected a BGR colour image")
        h, w = image.shape[:2]
        s = min(1.0, self.cfg.work_width / w)
        small = cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue, sat, val = (hsv[..., i].astype(np.float32) for i in range(3))
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        texture = cv2.blur(np.abs(cv2.Laplacian(grey, cv2.CV_32F)), (15, 15))

        labels = np.full(grey.shape, Label.OTHER, np.uint8)
        labels[(hue >= 35) & (hue <= 85) & (sat > 60) & (val > 30)] = Label.VEGETATION
        labels[(hue >= 90) & (hue <= 130) & (sat > 40) & (texture < 4)] = Label.WATER
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        return SegmentationResult(labels, np.full((h, w), self.CONFIDENCE, np.float32), self.name, True)


def make_segmenter(cfg: SegmentationConfig | None = None) -> Segmenter:
    """'auto': DeepLabV3 when its weights are cached locally, otherwise the placeholder."""
    cfg = cfg or SegmentationConfig()
    if cfg.backend == "mock":
        return MockSegmenter(cfg)
    if cfg.backend in ("deeplab", "auto"):
        from dronerecon.segmentation.deeplab import DeepLabSegmenter, weights_cached

        if cfg.backend == "deeplab" or weights_cached():
            return DeepLabSegmenter(cfg)
        return MockSegmenter(cfg)
    raise DataError(f"segmentation backend '{cfg.backend}' is not available; use 'auto', 'deeplab' or 'mock'")


# --------------------------------------------------------------------------- cache

def save_labels(result: SegmentationResult, path: str | Path) -> None:
    """Store labels as a lossless PNG (confidence is not cached)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(p.with_suffix(".png")), result.labels):
        raise DataError(f"could not write {p}")


def load_labels(path: str | Path, backend: str, is_mock: bool) -> SegmentationResult:
    p = Path(path).with_suffix(".png")
    labels = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise DataError(f"could not read label image {p}")
    return SegmentationResult(labels.astype(np.uint8), None, backend, is_mock)
