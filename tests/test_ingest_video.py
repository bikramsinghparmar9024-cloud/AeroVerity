import cv2
import numpy as np
import pytest

from dronerecon.config import PathsConfig
from dronerecon.ingest.video import VideoError, iter_frames, parse_ffprobe, probe_video

NO_FFPROBE = PathsConfig(ffprobe="definitely-not-ffprobe")


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    assert writer.isOpened()
    for i in range(12):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()
    return path


def test_probe_opencv_fallback(clip):
    info = probe_video(clip, NO_FFPROBE)
    assert (info.width, info.height, info.frame_count) == (64, 48, 12)
    assert info.fps == pytest.approx(10.0)
    assert info.codec == "MJPG"


def test_iter_frames_stride_and_time(clip):
    info = probe_video(clip, NO_FFPROBE)
    frames = list(iter_frames(info, stride=5))
    assert [f.index for f in frames] == [0, 5, 10]
    assert [f.t_video_s for f in frames] == pytest.approx([0.0, 0.5, 1.0])
    assert frames[1].image.shape == (48, 64, 3)
    assert abs(int(frames[1].image.mean()) - 100) < 10


def test_iter_frames_window(clip):
    info = probe_video(clip, NO_FFPROBE)
    assert [f.index for f in iter_frames(info, start=3, stop=6)] == [3, 4, 5]


def test_missing_video():
    with pytest.raises(VideoError, match="not found"):
        probe_video("nope.mp4")


def test_parse_ffprobe_ntsc_and_missing_frame_count():
    data = {
        "streams": [
            {"codec_type": "audio"},
            {"codec_type": "video", "codec_name": "h264", "width": 3840, "height": 2160,
             "avg_frame_rate": "30000/1001", "duration": "10.01"},
        ],
        "format": {"tags": {"creation_time": "2026-09-01T10:00:00.000000Z"}},
    }
    info = parse_ffprobe("x.mp4", data)
    assert info.fps == pytest.approx(29.97, abs=1e-2)
    assert info.frame_count == 300
    assert info.codec == "h264"
    assert info.creation_time_utc.startswith("2026-09-01")


def test_parse_ffprobe_no_video_stream():
    with pytest.raises(VideoError, match="no video stream"):
        parse_ffprobe("x.mp4", {"streams": [{"codec_type": "audio"}]})
