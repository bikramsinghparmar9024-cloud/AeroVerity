"""Calibrate bootstrap uncertainty against held-out views.

Held-out errors: in each bootstrap run some keyframes were left out. A point
re-triangulated without a keyframe is projected into that keyframe (using the
full-solution pose) and compared with where the keyframe actually saw it. The
pixel miss is converted to metres at the point's depth. That is an error the
run never had access to.

Calibration: if the bootstrap sigma were right, errors divided by sigma would
follow a known distribution (chi with `dims` degrees of freedom; 2 for image
errors). One scale factor k is fitted robustly from the median, and a
reliability diagram compares nominal against observed coverage before and
after scaling, plus RMS error per sigma bin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import chi

from aerometric.config import UncertaintyConfig
from aerometric.pose.types import Trajectory
from aerometric.types import DataError
from aerometric.uncertainty.bootstrap import RunDraw


@dataclass(frozen=True)
class HeldOutErrors:
    point_ids: np.ndarray  # M
    error_m: np.ndarray  # M, lateral miss in metres at the point's depth
    error_px: np.ndarray  # M


@dataclass(frozen=True)
class Calibration:
    scale: float  # multiply bootstrap sigma by this
    dims: int
    n: int
    levels: np.ndarray  # nominal coverage levels
    coverage_before: np.ndarray  # observed share of errors inside the nominal radius
    coverage_after: np.ndarray
    bin_sigma: np.ndarray  # mean calibrated sigma per bin
    bin_rms_error: np.ndarray  # RMS error per bin, per dimension (comparable to sigma)
    trusted: bool
    warnings: tuple[str, ...] = ()

    def apply(self, sigma: np.ndarray) -> np.ndarray:
        return sigma * self.scale


def held_out_errors(run_points: Sequence[np.ndarray], draws: Sequence[RunDraw], base: Trajectory,
                    tracks: Sequence[np.ndarray], keypoints: Sequence[np.ndarray]) -> HeldOutErrors:
    """run_points[r]: original tracks re-triangulated in run r (NaN where not possible)."""
    if len(run_points) != len(draws):
        raise DataError("calibration: one set of run points per draw is needed")
    K = base.intrinsics
    ids, err_m, err_px = [], [], []
    for pts, d in zip(run_points, draws):
        held = set(np.flatnonzero(~d.keep & base.valid).tolist())
        if not held:
            continue
        for pid, tr in enumerate(tracks):
            X = pts[pid]
            if not np.all(np.isfinite(X)):
                continue
            for c, k in tr:
                if c not in held:
                    continue
                p = base.poses[c]
                Xc = p.R_cw @ X + p.t_cw
                if Xc[2] <= 0:
                    continue
                uv = np.array([K.fx * Xc[0] / Xc[2] + K.cx, K.fy * Xc[1] / Xc[2] + K.cy])
                e = float(np.linalg.norm(uv - keypoints[c][k]))
                ids.append(pid)
                err_px.append(e)
                err_m.append(e * Xc[2] / K.fx)
    return HeldOutErrors(np.array(ids, int), np.array(err_m), np.array(err_px))


def calibrate(sigma: np.ndarray, error: np.ndarray, cfg: UncertaintyConfig | None = None,
              dims: int = 2) -> Calibration:
    """sigma: predicted per-axis 1-sigma for each error sample; error: radial error (same units)."""
    cfg = cfg or UncertaintyConfig()
    ok = np.isfinite(sigma) & np.isfinite(error) & (sigma > 0)
    s, e = sigma[ok], error[ok]
    levels = np.asarray(cfg.reliability_levels, float)
    if s.size == 0:
        raise DataError("calibration: no held-out errors with a predicted uncertainty")
    dist = chi(df=dims)
    z = e / s
    scale = float(np.median(z) / dist.median())
    radius = dist.ppf(levels)
    before = np.array([np.mean(z <= r) for r in radius])
    after = np.array([np.mean(z / scale <= r) for r in radius])

    edges = np.quantile(s, np.linspace(0, 1, cfg.reliability_bins + 1))
    edges[-1] += 1e-12
    b_sig, b_rms = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi)
        if m.any():
            b_sig.append(float(np.mean(s[m] * scale)))
            b_rms.append(float(np.sqrt(np.mean(e[m] ** 2) / dims)))

    warnings: list[str] = []
    trusted = s.size >= cfg.min_calibration_samples
    if not trusted:
        warnings.append(f"only {s.size} held-out errors; the uncertainty scale is a rough guide")
    if scale > cfg.max_scale_warning:
        warnings.append(f"the bootstrap understated uncertainty by about {scale:.1f}x; all sigmas were scaled up")
    elif scale < 1 / cfg.max_scale_warning:
        warnings.append(f"the bootstrap overstated uncertainty by about {1 / scale:.1f}x; all sigmas were scaled down")
    return Calibration(scale, dims, int(s.size), levels, before, after, np.array(b_sig), np.array(b_rms),
                       trusted, tuple(warnings))


DASHED = 'stroke-dasharray="5 4"'


def reliability_svg(cal: Calibration, width: int = 360, height: int = 300) -> str:
    """Self-contained SVG: nominal vs observed coverage, before (dashed) and after calibration."""
    pad = 44
    pw, ph = width - pad - 16, height - pad - 20

    def xy(x: float, y: float) -> str:
        return f"{pad + x * pw:.1f},{20 + (1 - y) * ph:.1f}"

    def line(ys, dash: str, color: str) -> str:
        pts = " ".join(xy(x, y) for x, y in zip(cal.levels, ys))
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" {dash}/>'

    ticks = "".join(
        f'<text x="{pad + t * pw:.1f}" y="{height - 22}" font-size="10" text-anchor="middle">{t:.1f}</text>'
        f'<text x="{pad - 6}" y="{20 + (1 - t) * ph + 3:.1f}" font-size="10" text-anchor="end">{t:.1f}</text>'
        for t in (0, 0.25, 0.5, 0.75, 1.0))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="system-ui, sans-serif" fill="#444">'
        f'<rect x="{pad}" y="20" width="{pw}" height="{ph}" fill="none" stroke="#ccc"/>'
        f'<line x1="{pad}" y1="{20 + ph}" x2="{pad + pw}" y2="20" stroke="#bbb" stroke-dasharray="2 3"/>'
        f'{line(cal.coverage_before, DASHED, "#a86400")}'
        f'{line(cal.coverage_after, "", "#2f6fdf")}'
        f'{ticks}'
        f'<text x="{pad + pw / 2}" y="{height - 6}" font-size="11" text-anchor="middle">Expected coverage</text>'
        f'<text x="12" y="{20 + ph / 2}" font-size="11" text-anchor="middle" '
        f'transform="rotate(-90 12 {20 + ph / 2})">Observed coverage</text>'
        f'<text x="{pad + 8}" y="34" font-size="11" fill="#2f6fdf">calibrated (×{cal.scale:.2f})</text>'
        f'<text x="{pad + 8}" y="48" font-size="11" fill="#a86400">raw bootstrap</text>'
        f'</svg>')
