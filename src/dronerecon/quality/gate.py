"""Pre-flight quality gate.

A fast heuristic check on a sample of frames plus the sync/GPS results. It
decides whether reconstruction is worth attempting and tells the operator why
not. The true coverage check happens later, after the first sparse model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import cv2
import numpy as np

from dronerecon.config import GateConfig
from dronerecon.sync.offset import OffsetEstimate
from dronerecon.sync.smoothing import SmoothedTrack
from dronerecon.types import DataError, Frame, VideoInfo

Status = Literal["pass", "warn", "fail"]
Decision = Literal["proceed", "proceed_with_warnings", "reject"]
BLOCK = 8


@dataclass(frozen=True)
class FrameMetrics:
    index: np.ndarray
    t_video_s: np.ndarray
    sharpness: np.ndarray  # variance of Laplacian
    brightness: np.ndarray  # mean grey level 0..255
    clipped: np.ndarray  # share of pixels at the dark/bright limits
    blockiness: np.ndarray  # ~1.0 clean, higher = visible compression blocks
    features: np.ndarray  # corner count


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str


@dataclass(frozen=True)
class GateReport:
    decision: Decision
    checks: tuple[CheckResult, ...]
    metrics: FrameMetrics

    def summary(self) -> str:
        head = {"proceed": "Ready to reconstruct.",
                "proceed_with_warnings": "Reconstruction can go ahead, but check the notes below.",
                "reject": "This flight is unlikely to give a usable model."}[self.decision]
        lines = [head] + [f"  [{c.status.upper()}] {c.name}: {c.message}" for c in self.checks]
        return "\n".join(lines)


# --------------------------------------------------------------------------- per-frame metrics

def _grey(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def sharpness(grey: np.ndarray) -> float:
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def blockiness(grey: np.ndarray, crop: int = 512) -> float:
    """Ratio of gradient on 8x8 block boundaries to gradient inside blocks (full resolution)."""
    h, w = grey.shape
    y0, x0 = max(0, (h - crop) // 2) // BLOCK * BLOCK, max(0, (w - crop) // 2) // BLOCK * BLOCK
    g = grey[y0:y0 + crop, x0:x0 + crop].astype(np.float64)

    def ratio(d: np.ndarray) -> float:
        on_edge = (np.arange(d.shape[1]) + 1) % BLOCK == 0
        inside = d[:, ~on_edge].mean()
        return float(d[:, on_edge].mean() / inside) if inside > 1e-6 else 1.0

    return (ratio(np.abs(np.diff(g, axis=1))) + ratio(np.abs(np.diff(g, axis=0)).T)) / 2


def frame_metrics(frames: Iterable[Frame], cfg: GateConfig | None = None) -> FrameMetrics:
    cfg = cfg or GateConfig()
    rows = []
    for f in frames:
        full = _grey(f.image)
        h, w = full.shape
        s = min(1.0, cfg.analysis_width / w)
        small = cv2.resize(full, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        clipped = float(np.mean((small <= cfg.dark_level) | (small >= cfg.bright_level)))
        corners = cv2.goodFeaturesToTrack(small, 2000, 0.01, 5)
        rows.append((f.index, f.t_video_s, sharpness(small), float(small.mean()), clipped,
                     blockiness(full), 0 if corners is None else len(corners)))
    if not rows:
        raise DataError("quality gate: no frames to inspect")
    cols = [np.array(c) for c in zip(*rows)]
    return FrameMetrics(cols[0].astype(int), *cols[1:6], cols[6].astype(int))


def sample_stride(video: VideoInfo, cfg: GateConfig | None = None) -> int:
    """Frame stride that gives roughly `sample_fps` inspected frames per second."""
    cfg = cfg or GateConfig()
    return max(1, round(video.fps / cfg.sample_fps))


# --------------------------------------------------------------------------- checks

def _fraction_check(name: str, bad: np.ndarray, cfg: GateConfig, what: str, advice: str) -> CheckResult:
    frac = float(bad.mean())
    if frac >= cfg.fail_bad_fraction:
        status: Status = "fail"
    elif frac >= cfg.warn_bad_fraction:
        status = "warn"
    else:
        return CheckResult(name, "pass", f"{frac:.0%} of sampled frames {what}.")
    return CheckResult(name, status, f"{frac:.0%} of sampled frames {what}. {advice}")


def _sync_checks(video: VideoInfo, offset: OffsetEstimate, track: SmoothedTrack,
                 cfg: GateConfig) -> list[CheckResult]:
    out: list[CheckResult] = []
    if offset.reliable:
        out.append(CheckResult("Telemetry sync", "pass",
                               f"Video and telemetry aligned (offset {offset.offset_s:+.2f} s, "
                               f"from {offset.method.replace('_', ' ')})."))
    else:
        out.append(CheckResult("Telemetry sync", "warn",
                               "Couldn't confirm the time offset between video and telemetry; "
                               f"using {offset.offset_s:+.2f} s. Positions may be shifted along the "
                               "flight path until the sparse model corrects them."))

    start, end = offset.offset_s, offset.offset_s + video.duration_s
    covered = max(0.0, min(end, track.t[-1]) - max(start, track.t[0])) / video.duration_s
    if covered < cfg.fail_sync_coverage:
        status: Status = "fail"
    elif covered < cfg.warn_sync_coverage:
        status = "warn"
    else:
        status = "pass"
    msg = f"Telemetry covers {covered:.0%} of the video."
    if status != "pass":
        msg += " Frames outside that window have no GPS position; check the log belongs to this clip."
    out.append(CheckResult("Telemetry coverage", status, msg))

    sigma = float(np.median(np.hypot(track.pos_std[:, 0], track.pos_std[:, 1])))
    status = "warn" if sigma > cfg.warn_gps_sigma_m else "pass"
    msg = f"Median horizontal GPS uncertainty is {sigma:.1f} m."
    if status == "warn":
        msg += " Absolute position of the model will be loose; RTK/PPK or known dimensions would help."
    out.append(CheckResult("GPS quality", status, msg))
    return out


def evaluate_gate(metrics: FrameMetrics, video: VideoInfo, offset: OffsetEstimate,
                  track: SmoothedTrack, cfg: GateConfig | None = None) -> GateReport:
    cfg = cfg or GateConfig()
    checks = [
        _fraction_check("Blur", metrics.sharpness < cfg.min_sharpness, cfg, "are blurry",
                        "Fly slower, use a faster shutter, or check the lens is clean."),
        _fraction_check("Exposure", (metrics.clipped > cfg.max_clipped_fraction)
                        | (metrics.brightness < cfg.min_brightness)
                        | (metrics.brightness > cfg.max_brightness), cfg,
                        "are badly exposed", "Large areas are crushed to black or blown out."),
        _fraction_check("Compression", metrics.blockiness > cfg.max_blockiness, cfg,
                        "show heavy compression blocks",
                        "Use the original recording rather than a re-encoded or streamed copy."),
        _fraction_check("Texture", metrics.features < cfg.min_features, cfg,
                        "have too little texture to match",
                        "Water, sky, snow or fog cannot be reconstructed from images alone."),
        *_sync_checks(video, offset, track, cfg),
    ]
    statuses = {c.status for c in checks}
    decision: Decision = ("reject" if "fail" in statuses
                          else "proceed_with_warnings" if "warn" in statuses else "proceed")
    return GateReport(decision, tuple(checks), metrics)
