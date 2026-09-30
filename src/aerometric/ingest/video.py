"""Video probing and frame decoding.

Metadata comes from ffprobe when it is installed (more reliable codec, frame
count and creation time); otherwise OpenCV is used. Frames are always decoded
with OpenCV. Frame timestamps assume constant frame rate: t = index / fps.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import cv2

from aerometric.config import PathsConfig
from aerometric.types import DataError, Frame, VideoInfo


class VideoError(DataError):
    """Raised when a video cannot be opened or probed."""


def _parse_rate(rate: str | None) -> float:
    if not rate or rate in ("0/0", "0"):
        return 0.0
    try:
        return float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return 0.0


def parse_ffprobe(path: str, data: dict[str, Any]) -> VideoInfo:
    """Build VideoInfo from `ffprobe -show_streams -show_format -of json` output."""
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise VideoError(f"{path}: no video stream found")
    s = streams[0]
    fmt = data.get("format", {})
    fps = _parse_rate(s.get("avg_frame_rate")) or _parse_rate(s.get("r_frame_rate"))
    if fps <= 0:
        raise VideoError(f"{path}: could not determine frame rate")
    frames = int(s.get("nb_frames") or 0)
    if frames <= 0:
        duration = float(s.get("duration") or fmt.get("duration") or 0)
        frames = round(duration * fps)
    tags = {**fmt.get("tags", {}), **s.get("tags", {})}
    return VideoInfo(
        path=path,
        width=int(s["width"]),
        height=int(s["height"]),
        fps=fps,
        frame_count=frames,
        codec=s.get("codec_name", "unknown"),
        creation_time_utc=tags.get("creation_time"),
    )


def _probe_ffprobe(path: str, ffprobe: str) -> VideoInfo | None:
    exe = shutil.which(ffprobe)
    if exe is None:
        return None
    cmd = [exe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise VideoError(f"{path}: ffprobe failed: {result.stderr.strip()}")
    return parse_ffprobe(path, json.loads(result.stdout))


def _probe_opencv(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoError(f"{path}: OpenCV could not open the video")
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ") or "unknown"
        return VideoInfo(
            path=path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            codec=codec,
        )
    finally:
        cap.release()


def probe_video(path: str | Path, paths: PathsConfig | None = None) -> VideoInfo:
    p = Path(path)
    if not p.is_file():
        raise VideoError(f"video not found: {p}")
    info = _probe_ffprobe(str(p), (paths or PathsConfig()).ffprobe)
    return info if info is not None else _probe_opencv(str(p))


def iter_frames(info: VideoInfo, stride: int = 1, start: int = 0,
                stop: int | None = None) -> Iterator[Frame]:
    """Yield every `stride`-th frame in [start, stop) as BGR uint8."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    cap = cv2.VideoCapture(info.path)
    if not cap.isOpened():
        raise VideoError(f"{info.path}: OpenCV could not open the video")
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        index = start
        while stop is None or index < stop:
            if not cap.grab():
                break
            if (index - start) % stride == 0:
                ok, image = cap.retrieve()
                if not ok:
                    raise VideoError(f"{info.path}: failed to decode frame {index}")
                yield Frame(index=index, t_video_s=index / info.fps, image=image)
            index += 1
    finally:
        cap.release()
