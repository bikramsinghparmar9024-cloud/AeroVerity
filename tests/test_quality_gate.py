import cv2
import numpy as np
import pytest

from dronerecon.geo import LocalFrame
from dronerecon.quality.gate import (
    blockiness,
    evaluate_gate,
    frame_metrics,
    sample_stride,
)
from dronerecon.sync.offset import OffsetEstimate
from dronerecon.sync.smoothing import SmoothedTrack
from dronerecon.types import Frame, VideoInfo

VIDEO = VideoInfo("v.mp4", 960, 540, 30.0, 300)  # 10 s
GOOD_SYNC = OffsetEstimate(0.3, 0.9, True, "gps_speed")


def scene(seed=0, h=540, w=960):
    rng = np.random.default_rng(seed)
    small = (rng.random((h // 6, w // 6)) * 200 + 28).astype(np.uint8)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def blurred(img):
    return cv2.GaussianBlur(img, (0, 0), 6)


def dark(img):
    return (img * 0.03).astype(np.uint8)


def jpeg_blocky(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 4])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def flat(img):
    return np.full_like(img, 120)


def frames(n=10, spoil=None, spoil_every=1):
    out = []
    for i in range(n):
        img = scene(i)
        if spoil is not None and i % spoil_every == 0:
            img = spoil(img)
        out.append(Frame(i * 15, i * 0.5, img))
    return out


def track(t0=0.0, t1=11.0, sigma=0.8):
    t = np.linspace(t0, t1, 50)
    enu = np.zeros((50, 3))
    return SmoothedTrack(t, enu, enu, np.full((50, 3), sigma), LocalFrame(28.6, 77.2, 100),
                         "gnss", np.zeros(50, bool))


def run(frs, offset=GOOD_SYNC, trk=None):
    return evaluate_gate(frame_metrics(frs), VIDEO, offset, trk or track())


def by_name(report):
    return {c.name: c for c in report.checks}


def test_clean_flight_proceeds():
    rep = run(frames())
    assert rep.decision == "proceed", rep.summary()
    assert rep.summary().startswith("Ready to reconstruct.")


@pytest.mark.parametrize("spoil,check", [
    (blurred, "Blur"), (dark, "Exposure"), (jpeg_blocky, "Compression"), (flat, "Texture"),
])
def test_each_image_check_fails_on_its_defect(spoil, check):
    rep = run(frames(spoil=spoil))
    assert by_name(rep)[check].status == "fail", rep.summary()
    assert rep.decision == "reject"


def test_partial_blur_only_warns():
    rep = run(frames(spoil=blurred, spoil_every=3))  # 4 of 10 frames
    assert by_name(rep)["Blur"].status == "warn"
    assert rep.decision == "proceed_with_warnings"


def test_blockiness_metric_separates_clean_and_compressed():
    clean = cv2.cvtColor(scene(), cv2.COLOR_BGR2GRAY)
    blocky = cv2.cvtColor(jpeg_blocky(scene()), cv2.COLOR_BGR2GRAY)
    assert blockiness(clean) < 1.15 < 1.35 < blockiness(blocky)


def test_sync_problems():
    rep = run(frames(), offset=OffsetEstimate(0.0, 0.2, False, "gps_speed"),
              trk=track(0.0, 4.0, sigma=9.0))
    c = by_name(rep)
    assert c["Telemetry sync"].status == "warn"
    assert c["Telemetry coverage"].status == "fail"  # 4 s of a 10 s clip
    assert c["GPS quality"].status == "warn"
    assert "RTK/PPK" in c["GPS quality"].message
    assert rep.decision == "reject"


def test_sample_stride():
    assert sample_stride(VIDEO) == 15
    assert sample_stride(VideoInfo("v", 10, 10, 1.0, 5)) == 1
