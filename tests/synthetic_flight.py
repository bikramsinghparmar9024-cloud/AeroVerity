"""A small synthetic drone flight: rendered video + DJI-style SRT + intrinsics, with known truth.

Flat textured ground, camera tilted 10 degrees forward at 50 m, flying east with
a varying speed so that video/telemetry sync has something to lock onto. The SRT
clock runs `offset_s` ahead of the video (t_telemetry = t_video + offset_s).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from aerometric.geo import LocalFrame
from aerometric.pose.types import CameraPose

W, H, F = 640, 480, 500.0
FPS = 15.0
ALT = 50.0
GSD = 0.1  # metres per texture pixel
FRAME = LocalFrame(28.6, 77.2, 200.0)  # ground level; GPS altitude = 250 m


@dataclass(frozen=True)
class SyntheticFlight:
    video: Path
    srt: Path
    intrinsics: Path
    offset_s: float
    centers_at: object  # callable t_video -> ENU camera centre


def speed(t):
    return 7 + 3 * np.sin(0.8 * t) + 2 * np.exp(-((t - 5) ** 2))


def _positions(t):
    dt = 0.01
    grid = np.arange(0, t.max() + dt, dt)
    x = np.concatenate([[0], np.cumsum(speed(grid[:-1]) * dt)]) - 40
    e = np.interp(t, grid, x)
    return np.column_stack([e, 3 * np.sin(t / 3), np.full_like(t, ALT)])


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    w, h = int(260 / GSD), int(120 / GSD)  # 260 m x 120 m of ground
    img = cv2.resize((rng.random((h // 20, w // 20, 3)) * 200 + 30).astype(np.uint8), (w, h),
                     interpolation=cv2.INTER_CUBIC)
    for _ in range(6000):
        x, y = rng.integers(0, w - 60, 2)
        col = tuple(int(c) for c in rng.integers(0, 255, 3))
        if rng.random() < 0.6:
            cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(6, 60)), int(y + rng.integers(6, 60))), col, -1)
        else:
            cv2.circle(img, (int(x), int(y)), int(rng.integers(3, 25)), col, -1)
    return img  # covers E in [-80, 180], N in [-60, 60]


def _pose(center, t):
    tilt = np.radians(10)
    yaw = np.radians(2 * np.sin(t))
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])  # east-facing camera: image right = south
    z = np.cos(tilt) * np.array([0, 0, -1.0]) + np.sin(tilt) * fwd
    y = np.cross(z, right)
    R = np.array([right, y, z])
    return CameraPose.from_center(R, center)


def _render(tex, pose):
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    rays = np.column_stack([(u.ravel() - W / 2) / F, (v.ravel() - H / 2) / F, np.ones(u.size)]) @ pose.R_cw
    C = pose.center
    s = -C[2] / rays[:, 2]
    P = C + rays * s[:, None]
    mx = ((P[:, 0] + 80) / GSD).reshape(H, W).astype(np.float32)
    my = ((60 - P[:, 1]) / GSD).reshape(H, W).astype(np.float32)
    return cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def make_flight(folder: Path, duration_s=10.0, offset_s=0.3, gps_noise_m=0.4, seed=0) -> SyntheticFlight:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tex = _texture(seed)
    n = int(duration_s * FPS)
    t = np.arange(n) / FPS
    C = _positions(t)

    video = folder / "flight.avi"
    wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(n):
        wr.write(_render(tex, _pose(C[i], t[i])))
    wr.release()

    # SRT at the video frame rate, stamped on the telemetry clock
    t_tel = np.arange(0, duration_s + 1.0, 1 / FPS)
    # GPS error drifts slowly (as real receivers do) rather than jumping every sample
    k = int(FPS)  # ~1 s smoothing
    white = rng.normal(0, 1, (len(t_tel) + k, 3))
    drift = np.stack([np.convolve(white[:, i], np.ones(k) / np.sqrt(k), "valid")[:len(t_tel)] for i in range(3)], 1)
    Ct = _positions(np.clip(t_tel - offset_s, 0, None)) + drift * [gps_noise_m, gps_noise_m, 2 * gps_noise_m]
    lat, lon, alt = FRAME.to_lla(Ct)
    blocks = []
    for i, (tt, la, lo, al) in enumerate(zip(t_tel, lat, lon, alt)):
        a, b = tt, tt + 1 / FPS
        stamp = lambda s: f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d},{int(round(s % 1 * 1000)):03d}"  # noqa: E731
        blocks.append(f"{i + 1}\n{stamp(a)} --> {stamp(b)}\n<font size=\"28\">FrameCnt: {i + 1}\n"
                      f"[latitude: {la:.7f}] [longitude: {lo:.7f}] [rel_alt: {al - 200:.3f} abs_alt: {al:.3f}] "
                      f"[gb_yaw: {90 - 2 * np.sin(max(tt - offset_s, 0)):.1f} gb_pitch: -80.0 gb_roll: 0.0] </font>\n")
    srt = folder / "flight.srt"
    srt.write_text("\n".join(blocks))

    intr = folder / "camera.json"
    intr.write_text(json.dumps({"width": W, "height": H, "fx": F, "fy": F, "cx": W / 2, "cy": H / 2}))
    return SyntheticFlight(video, srt, intr, offset_s, lambda tv: _positions(np.atleast_1d(tv)))
