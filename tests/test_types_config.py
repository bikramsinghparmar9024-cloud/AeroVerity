import json

import numpy as np
import pytest

from dronerecon.config import ConfigError, PipelineConfig, load_config
from dronerecon.types import (
    CameraIntrinsics,
    DataError,
    FlightRecord,
    GpsTrack,
    ImuSeries,
    VideoInfo,
)


def _gps(n=5, source="gnss"):
    t = np.arange(n, dtype=float)
    return GpsTrack(t, np.full(n, 28.6), np.full(n, 77.2), np.full(n, 100.0), source=source)


def test_gps_track_valid_and_readonly():
    g = _gps()
    assert len(g) == 5
    with pytest.raises(ValueError):
        g.lat[0] = 1.0


@pytest.mark.parametrize("t", [[0, 1, 1], [0, 2, 1], [0, np.nan, 2]])
def test_gps_track_rejects_bad_time(t):
    with pytest.raises(DataError):
        GpsTrack(t, [0, 0, 0], [0, 0, 0], [0, 0, 0])


def test_gps_track_rejects_length_mismatch_and_range():
    with pytest.raises(DataError, match="length"):
        GpsTrack([0, 1], [0], [0, 0], [0, 0])
    with pytest.raises(DataError, match="range"):
        GpsTrack([0, 1], [95, 0], [0, 0], [0, 0])


def test_imu_shape_checked():
    with pytest.raises(DataError, match="columns"):
        ImuSeries([0, 1], np.zeros((2, 2)))


def test_intrinsics_from_fov():
    k = CameraIntrinsics.from_fov(1920, 1080, 90.0)
    assert k.fx == pytest.approx(960.0)
    assert k.source == "default"
    assert k.K.shape == (3, 3)


def test_flight_record_checks_intrinsics_size_and_rtk_source():
    video = VideoInfo("v.mp4", 1920, 1080, 30.0, 300)
    assert video.duration_s == pytest.approx(10.0)
    with pytest.raises(DataError, match="does not match"):
        FlightRecord(video, _gps(), CameraIntrinsics.from_fov(1280, 720, 84))
    with pytest.raises(DataError, match="rtk"):
        FlightRecord(video, _gps(), CameraIntrinsics.from_fov(1920, 1080, 84), rtk=_gps())
    rec = FlightRecord(video, _gps(), CameraIntrinsics.from_fov(1920, 1080, 84), rtk=_gps(source="rtk"))
    assert rec.rtk is not None


def test_load_config_defaults_and_overrides(tmp_path):
    assert load_config() == PipelineConfig()
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"ingest": {"frame_stride": 3}}))
    cfg = load_config(p)
    assert cfg.ingest.frame_stride == 3
    assert cfg.ingest.default_hfov_deg == 84.0


def test_load_config_errors(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"ingest": {"bogus": 1}}))
    with pytest.raises(ConfigError, match="bogus"):
        load_config(p)
    p.write_text(json.dumps({"ingest": {"frame_stride": 0}}))
    with pytest.raises(ConfigError, match="frame_stride"):
        load_config(p)
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.json")


def test_repo_default_config_loads():
    from pathlib import Path

    cfg = load_config(Path(__file__).parents[1] / "configs" / "default.json")
    assert cfg == PipelineConfig()
