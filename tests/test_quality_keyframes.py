import cv2
import numpy as np
import pytest

from dronerecon.config import KeyframeConfig
from dronerecon.geo import LocalFrame
from dronerecon.quality.keyframes import select_keyframes
from dronerecon.sync.smoothing import SmoothedTrack
from dronerecon.types import AttitudeSeries, Frame

W, H, FPS = 480, 270, 10.0


def texture(w=3000, h=400, seed=0):
    rng = np.random.default_rng(seed)
    small = (rng.random((h // 4, w // 4)) * 200 + 28).astype(np.uint8)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def pan(n=120, px_per_frame=6.0, blur_every=0):
    """Camera sliding east over a textured ground at constant speed."""
    tex = texture()
    out = []
    for i in range(n):
        M = np.float32([[1, 0, -i * px_per_frame], [0, 1, -60]])
        img = cv2.warpAffine(tex, M, (W, H), flags=cv2.INTER_LINEAR)
        if blur_every and i % blur_every == 1:
            img = cv2.GaussianBlur(img, (0, 0), 5)
        out.append(Frame(i, i / FPS, img))
    return out


def static(n=60):
    img = cv2.warpAffine(texture(), np.float32([[1, 0, 0], [0, 1, -60]]), (W, H))
    return [Frame(i, i / FPS, img) for i in range(n)]


def test_spacing_follows_target_parallax():
    ks = select_keyframes(pan())
    assert ks.keyframes[0].reason == "first" and ks.keyframes[0].index == 0
    assert ks.keyframes[-1].reason == "last"
    inner = [k.parallax for k in ks.keyframes[1:-1]]
    cfg = KeyframeConfig()
    assert all(cfg.candidate_parallax - 0.01 <= p <= cfg.target_parallax + 0.02 for p in inner)
    # 6 px/frame at 480 px width: ~5-7 frames per keyframe
    gaps = np.diff(ks.indices)
    assert 4 <= np.median(gaps) <= 8
    assert not ks.warnings


def test_blurry_frames_are_skipped():
    frames = pan(blur_every=3)
    blurry = {f.index for f in frames if f.index % 3 == 1}
    ks = select_keyframes(frames)
    assert not blurry & set(ks.indices)
    assert not any(k.forced for k in ks.keyframes)


def test_static_camera_gives_few_keyframes_and_a_warning():
    ks = select_keyframes(static())
    assert ks.indices == [0]
    assert any("fewer than 3" in w for w in ks.warnings)


def test_baseline_trigger_uses_gps_track():
    t = np.linspace(-1, 7, 81)
    enu = np.column_stack([10.0 * t, np.zeros_like(t), np.zeros_like(t)])  # 10 m/s east
    track = SmoothedTrack(t, enu, np.zeros_like(enu), np.ones_like(enu), LocalFrame(0, 0, 0),
                          "gnss", np.zeros(len(t), bool))
    ks = select_keyframes(static(), KeyframeConfig(max_baseline_m=12.0), track=track)
    assert ks.keyframes[-1].reason == "last"
    inner = ks.keyframes[1:-1]
    assert inner and all(k.reason == "baseline" for k in inner)
    assert all(k.baseline_m == pytest.approx(12.0, abs=1.1) for k in inner)


def test_rotation_trigger_uses_attitude():
    t = np.linspace(-1, 7, 81)
    yaw = (170 + 10.0 * t + 180) % 360 - 180  # 10 deg/s, wraps through ±180
    att = AttitudeSeries(t, np.column_stack([yaw, np.full_like(t, -90), np.zeros_like(t)]))
    ks = select_keyframes(static(), attitude=att)
    inner = ks.keyframes[1:-1]
    assert len(inner) >= 3 and all(k.reason == "rotation" for k in inner)
    assert np.diff([k.t_video_s for k in ks.keyframes[:-1]]) == pytest.approx(1.5, abs=0.11)
