"""Keyframe selection by sharpness, parallax and viewpoint change.

Features detected on the last keyframe are tracked frame to frame. Once the
median image shift passes `candidate_parallax`, sharp frames become candidates;
when a trigger fires (enough parallax, overlap running out, baseline or yaw
limit), the sharpest candidate becomes the next keyframe and tracking restarts
from it. Frames after the chosen one are replayed against the new keyframe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import cv2
import numpy as np

from dronerecon.config import KeyframeConfig
from dronerecon.quality.gate import sharpness
from dronerecon.sync.smoothing import SmoothedTrack
from dronerecon.types import AttitudeSeries, DataError, Frame


@dataclass(frozen=True)
class Keyframe:
    index: int
    t_video_s: float
    sharpness: float
    parallax: float  # median feature shift from previous keyframe, share of width
    baseline_m: float  # distance from previous keyframe (nan without a GPS track)
    reason: str  # first | parallax | overlap | baseline | rotation | last
    forced: bool = False  # no sharp candidate was available when the trigger fired


@dataclass(frozen=True)
class KeyframeSet:
    keyframes: tuple[Keyframe, ...]
    frames_seen: int
    warnings: tuple[str, ...] = ()

    @property
    def indices(self) -> list[int]:
        return [k.index for k in self.keyframes]


@dataclass
class _Obs:
    index: int
    t: float
    grey: np.ndarray
    sharp: float
    parallax: float = 0.0
    baseline: float = float("nan")
    dyaw: float = 0.0
    tracks: int = 0
    survival: float = 1.0
    progress: float = 0.0  # 1.0 = a parallax, baseline or yaw limit has been reached


@dataclass
class _Selector:
    cfg: KeyframeConfig
    position: Callable[[float], np.ndarray] | None
    yaw: Callable[[float], float] | None
    keyframes: list[Keyframe] = field(default_factory=list)
    kf: _Obs | None = None
    recent: list[_Obs] = field(default_factory=list)
    pts_kf: np.ndarray = field(default_factory=lambda: np.empty((0, 1, 2), np.float32))
    pts_cur: np.ndarray = field(default_factory=lambda: np.empty((0, 1, 2), np.float32))
    prev: np.ndarray | None = None
    n0: int = 0

    def push(self, obs: _Obs) -> None:
        if self.kf is None:
            if obs.sharp >= self.cfg.min_sharpness:
                self._set_keyframe(obs, "first", False)
            return
        self._track(obs)
        self.recent.append(obs)
        if (reason := self._trigger(obs)):
            self._choose(reason)

    def finish(self) -> None:
        half = self.cfg.candidate_parallax / self.cfg.target_parallax / 2
        eligible = [o for o in self.recent if o.progress >= half and o.sharp >= self.cfg.min_sharpness]
        if eligible:
            self._record(max(eligible, key=lambda o: (o.sharp, o.index)), "last", False)

    # ------------------------------------------------------------------ internals

    def _set_keyframe(self, obs: _Obs, reason: str, forced: bool) -> None:
        self._record(obs, reason, forced)
        pts = cv2.goodFeaturesToTrack(obs.grey, self.cfg.max_corners, 0.01, 7)
        self.pts_kf = pts if pts is not None else np.empty((0, 1, 2), np.float32)
        self.pts_cur, self.prev, self.n0 = self.pts_kf.copy(), obs.grey, len(self.pts_kf)
        self.kf, self.recent = obs, []

    def _record(self, obs: _Obs, reason: str, forced: bool) -> None:
        self.keyframes.append(Keyframe(obs.index, obs.t, obs.sharp, obs.parallax if self.keyframes else 0.0,
                                       obs.baseline if self.keyframes else 0.0, reason, forced))

    def _track(self, obs: _Obs) -> None:
        assert self.kf is not None and self.prev is not None
        if len(self.pts_cur):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, obs.grey, self.pts_cur, None)
            ok = st.ravel() == 1
            self.pts_cur, self.pts_kf = nxt[ok], self.pts_kf[ok]
        self.prev = obs.grey
        obs.tracks = len(self.pts_cur)
        obs.survival = obs.tracks / self.n0 if self.n0 else 0.0
        shift = np.linalg.norm((self.pts_cur - self.pts_kf).reshape(-1, 2), axis=1)
        obs.parallax = float(np.median(shift)) / obs.grey.shape[1] if len(shift) else float("inf")
        if self.position is not None:
            obs.baseline = float(np.linalg.norm(self.position(obs.t) - self.position(self.kf.t)))
        if self.yaw is not None:
            obs.dyaw = abs((self.yaw(obs.t) - self.yaw(self.kf.t) + 180) % 360 - 180)
        c = self.cfg
        parts = [min(obs.parallax, 1e6) / c.target_parallax, obs.dyaw / c.max_yaw_deg]
        if c.max_baseline_m > 0 and np.isfinite(obs.baseline):
            parts.append(obs.baseline / c.max_baseline_m)
        obs.progress = max(parts)

    def _trigger(self, obs: _Obs) -> str | None:
        c = self.cfg
        if obs.survival < c.min_survival or obs.tracks < c.min_tracks:
            return "overlap"
        if obs.parallax >= c.target_parallax:
            return "parallax"
        if c.max_baseline_m > 0 and obs.baseline >= c.max_baseline_m:
            return "baseline"
        if obs.dyaw >= c.max_yaw_deg:
            return "rotation"
        return None

    def _choose(self, reason: str) -> None:
        c = self.cfg
        min_progress = c.candidate_parallax / c.target_parallax
        good = [o for o in self.recent if o.sharp >= c.min_sharpness and o.progress >= min_progress]
        pick = max(good, key=lambda o: (o.sharp, o.index)) if good else self.recent[-1]
        later = self.recent[self.recent.index(pick) + 1:]
        self._set_keyframe(pick, reason, forced=not good)
        for o in later:
            self.push(o)


def select_keyframes(frames: Iterable[Frame], cfg: KeyframeConfig | None = None,
                     track: SmoothedTrack | None = None, offset_s: float = 0.0,
                     attitude: AttitudeSeries | None = None) -> KeyframeSet:
    """Choose keyframes from decoded frames (video clock). `offset_s` maps video to telemetry time."""
    cfg = cfg or KeyframeConfig()
    position = None
    if track is not None:
        position = lambda t: track.interpolate(t + offset_s)[0][0]  # noqa: E731
    yaw = None
    if attitude is not None:
        unwrapped = np.degrees(np.unwrap(np.radians(attitude.ypr_deg[:, 0])))
        yaw = lambda t: float(np.interp(t + offset_s, attitude.t, unwrapped))  # noqa: E731

    sel = _Selector(cfg, position, yaw)
    seen = 0
    for f in frames:
        seen += 1
        grey = cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY) if f.image.ndim == 3 else f.image
        h, w = grey.shape
        s = min(1.0, cfg.work_width / w)
        small = cv2.resize(grey, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        sel.push(_Obs(f.index, f.t_video_s, small, sharpness(small)))
    sel.finish()

    if not sel.keyframes:
        raise DataError("keyframes: no frame was sharp enough to start from")
    warnings = []
    if (forced := sum(k.forced for k in sel.keyframes)):
        warnings.append(f"{forced} keyframe(s) had to be taken from blurry or low-parallax frames "
                        "to keep overlap; expect weaker geometry there")
    if len(sel.keyframes) < 3:
        warnings.append("fewer than 3 keyframes; the clip may be too short or the camera barely moved")
    return KeyframeSet(tuple(sel.keyframes), seen, tuple(warnings))
