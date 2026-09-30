import json

import cv2
import numpy as np
import pytest

from dronerecon.config import PathsConfig, PipelineConfig
from dronerecon.ingest.flight import FlightInputs, ingest_flight
from dronerecon.ingest.intrinsics import load_intrinsics
from dronerecon.ingest.rtk import parse_rtklib_pos
from dronerecon.types import DataError

POS = """% program   : RTKPOST ver.2.4.3
%  GPST                  latitude(deg) longitude(deg)  height(m)   Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m) age(s)  ratio
2026/09/01 10:00:00.000   28.613900000   77.209000000   260.5000   1  12   0.0300   0.0400   0.0500   0.0000   0.0000   0.0000   0.00    9.9
2026/09/01 10:00:00.200   28.613901000   77.209001000   260.5100   2  12   0.2000   0.2000   0.3000   0.0000   0.0000   0.0000   0.00    1.2
2026/09/01 10:00:00.400   28.613902000   77.209002000   260.5200   5  12   2.0000   2.0000   3.0000   0.0000   0.0000   0.0000   0.00    0.0
2026/09/01 10:00:00.600   28.613903000   77.209003000   260.5300   1  12   0.0300   0.0400   0.0500   0.0000   0.0000   0.0000   0.00    9.9
"""


def test_rtklib_pos_quality_filter_and_accuracy(tmp_path):
    p = tmp_path / "flight.pos"
    p.write_text(POS)
    r = parse_rtklib_pos(p)
    assert r.track.source == "ppk"
    assert r.track.t == pytest.approx([0.0, 0.2, 0.6])
    assert r.track.h_acc_m[0] == pytest.approx(0.05)
    assert r.fix_ratio == pytest.approx(2 / 3)
    assert r.warnings  # one single-point epoch skipped
    assert r.start_gpst_s > 1.4e9


def test_rtklib_week_tow_and_bad_line(tmp_path):
    p = tmp_path / "w.pos"
    p.write_text("2385 360000.0 28.6 77.2 260 1 10 0.01 0.01 0.02\n"
                 "2385 360000.2 28.6 77.2 260 1 10 0.01 0.01 0.02\n")
    assert parse_rtklib_pos(p, source="rtk").start_gpst_s == pytest.approx(2385 * 604800 + 360000)
    p.write_text("2385 360000.0 -1234567.1 5678901.2 3000000.3\n")
    with pytest.raises(DataError, match="unrecognised"):
        parse_rtklib_pos(p)


def test_intrinsics_layouts_and_scaling(tmp_path):
    p = tmp_path / "k.json"
    p.write_text(json.dumps({"image_width": 3840, "image_height": 2160,
                             "camera_matrix": [[2800, 0, 1920], [0, 2800, 1080], [0, 0, 1]],
                             "dist_coeffs": [[0.1, -0.05, 0, 0, 0.01]]}))
    k = load_intrinsics(p, 1920, 1080)
    assert (k.fx, k.cx, k.model, len(k.dist)) == (1400, 960, "opencv", 5)

    p.write_text(json.dumps({"fx": 1000, "fy": 1000, "cx": 960, "cy": 540}))
    assert load_intrinsics(p, 1920, 1080).model == "pinhole"

    p.write_text(json.dumps({"hfov_deg": 90}))
    assert load_intrinsics(p, 1920, 1080).fx == pytest.approx(960)

    p.write_text(json.dumps({"width": 1440, "height": 1080, "fx": 1, "fy": 1, "cx": 1, "cy": 1}))
    with pytest.raises(DataError, match="aspect"):
        load_intrinsics(p, 1920, 1080)


@pytest.fixture
def flight_files(tmp_path):
    video = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for _ in range(20):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    tele = tmp_path / "log.csv"
    rows = "\n".join(f"{i * 0.1:.1f},28.6139,{77.209 + i * 1e-5},100" for i in range(21))
    tele.write_text("time_s,latitude,longitude,alt_m\n" + rows + "\n")
    return video, tele


def test_ingest_flight_minimal(flight_files):
    video, tele = flight_files
    cfg = PipelineConfig(paths=PathsConfig(ffprobe="no-ffprobe"))
    rec = ingest_flight(FlightInputs(video, tele), cfg)
    assert rec.video.frame_count == 20 and len(rec.gps) == 21
    assert rec.intrinsics.source == "default"
    assert any("intrinsics" in w for w in rec.warnings)
    assert not any("check that the files" in w for w in rec.warnings)


def test_ingest_flight_with_all_optional_inputs(flight_files, tmp_path):
    video, tele = flight_files
    (tmp_path / "imu.csv").write_text("t,gx,gy,gz\n0,0,0,0\n0.01,0,0,0\n")
    (tmp_path / "baro.csv").write_text("t,baro_alt\n0,1\n0.1,1.1\n")
    (tmp_path / "f.pos").write_text(POS)
    (tmp_path / "k.json").write_text(json.dumps({"hfov_deg": 70}))
    cfg = PipelineConfig(paths=PathsConfig(ffprobe="no-ffprobe"))
    rec = ingest_flight(FlightInputs(video, tele, tmp_path / "imu.csv", tmp_path / "baro.csv",
                                     tmp_path / "f.pos", tmp_path / "k.json"), cfg)
    assert rec.imu is not None and rec.baro is not None
    assert rec.rtk.source == "ppk"
    assert rec.intrinsics.source == "default"  # hfov prior, not a real calibration
