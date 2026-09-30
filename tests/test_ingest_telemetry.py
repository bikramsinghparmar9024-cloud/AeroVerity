import numpy as np
import pytest

from dronerecon.config import TelemetryConfig
from dronerecon.ingest.telemetry import (
    load_telemetry,
    parse_baro_csv,
    parse_dji_srt,
    parse_imu_csv,
)
from dronerecon.types import DataError

MODERN_SRT = """1
00:00:00,000 --> 00:00:00,033
<font size="28">FrameCnt: 1, DiffTime: 33ms
2026-09-01 10:00:00.000
[iso: 100] [shutter: 1/500.0] [fnum: 2.8] [focal_len: 24.00] [latitude: 28.613900] [longitude: 77.209000] [rel_alt: 50.000 abs_alt: 260.500] [gb_yaw: 10.0 gb_pitch: -90.0 gb_roll: 0.0] </font>

2
00:00:00,033 --> 00:00:00,066
<font size="28">FrameCnt: 2, DiffTime: 33ms
[latitude: 0.000000] [longitude: 0.000000] [rel_alt: 50.000 abs_alt: 260.500] [gb_yaw: 10.0 gb_pitch: -90.0 gb_roll: 0.0] </font>

3
00:00:00,066 --> 00:00:00,100
<font size="28">FrameCnt: 3, DiffTime: 33ms
[latitude: 28.613910] [longtitude: 77.209010] [rel_alt: 50.200 abs_alt: 260.700] [gb_yaw: 10.5 gb_pitch: -90.0 gb_roll: 0.1] </font>

4
00:00:00,100 --> 00:00:00,133
[latitude: 28.613920] [longitude: 77.209020] [rel_alt: 50.400 abs_alt: 260.900] [gb_yaw: 11.0 gb_pitch: -90.0 gb_roll: 0.1]
"""

LEGACY_SRT = """1
00:00:01,000 --> 00:00:02,000
HOME(77.2000,28.6000) 2016.05.01 10:00:00
GPS(77.2090,28.6139,15) BAROMETER:30.5

2
00:00:02,000 --> 00:00:03,000
GPS(77.2091,28.6140,15) BAROMETER:31.0
"""


def test_modern_srt(tmp_path):
    p = tmp_path / "DJI_0001.SRT"
    p.write_text(MODERN_SRT)
    r = parse_dji_srt(p)
    assert len(r.gps) == 3
    assert r.gps.t == pytest.approx([0.0, 0.066, 0.1])
    assert r.gps.lon[1] == pytest.approx(77.20901)  # 'longtitude' typo handled
    assert r.gps.alt_m[0] == pytest.approx(260.5)
    assert r.baro.alt_m[-1] == pytest.approx(50.4)
    assert r.attitude.ypr_deg[1] == pytest.approx([10.5, -90.0, 0.1])
    assert any("without a valid GPS fix" in w for w in r.warnings)


def test_legacy_srt_lon_lat_order(tmp_path):
    p = tmp_path / "old.srt"
    p.write_text(LEGACY_SRT)
    r = load_telemetry(p)
    assert r.gps.lat[0] == pytest.approx(28.6139)
    assert r.gps.lon[0] == pytest.approx(77.2090)
    assert r.baro.alt_m.tolist() == [30.5, 31.0]
    assert r.attitude is None


def test_srt_without_gps(tmp_path):
    p = tmp_path / "x.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:01,000\n[iso: 100]\n")
    with pytest.raises(DataError, match="no GPS"):
        parse_dji_srt(p)


def test_airdata_style_csv_ms_and_feet(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text(
        "time(millisecond),latitude,longitude,altitude_above_seaLevel(feet)\n"
        "1000,28.6139,77.2090,1000\n"
        "1100,28.6140,77.2091,1001\n"
        "1100,28.6140,77.2091,1001\n"
        "1200,28.6141,77.2092,n/a\n"
        "1300,28.6142,77.2093,1002\n"
    )
    r = load_telemetry(p)
    assert r.gps.t == pytest.approx([0.0, 0.1, 0.3])
    assert r.gps.alt_m[0] == pytest.approx(304.8)
    assert len(r.warnings) == 2  # one no-fix row, one duplicate timestamp


def test_csv_column_override_and_missing(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text("sec,y,x,z\n0,28.6,77.2,100\n1,28.61,77.21,101\n")
    cfg = TelemetryConfig(time_col="sec", lat_col="y", lon_col="x", alt_col="z")
    assert len(load_telemetry(p, cfg).gps) == 2
    with pytest.raises(DataError, match="no time column"):
        load_telemetry(p)


def test_imu_csv_deg_to_rad_with_accel(tmp_path):
    p = tmp_path / "imu.csv"
    p.write_text("t,gx,gy,gz,ax,ay,az\n0,180,0,0,0,0,9.81\n0.005,0,90,0,0,0,9.8\n")
    imu = parse_imu_csv(p, TelemetryConfig(gyro_unit="deg/s"))
    assert imu.gyro_rad_s[0, 0] == pytest.approx(np.pi)
    assert imu.accel_m_s2.shape == (2, 3)


def test_imu_csv_gyro_only(tmp_path):
    p = tmp_path / "imu.csv"
    p.write_text("time_s,gyro_x,gyro_y,gyro_z\n0,0.1,0,0\n0.01,0.2,0,0\n")
    imu = parse_imu_csv(p)
    assert imu.accel_m_s2 is None and len(imu.t) == 2


def test_baro_csv(tmp_path):
    p = tmp_path / "baro.csv"
    p.write_text("t,baro_alt\n0,10\n0.1,10.2\n")
    assert parse_baro_csv(p).alt_m.tolist() == [10.0, 10.2]


def test_unsupported_extension(tmp_path):
    with pytest.raises(DataError, match="unsupported"):
        load_telemetry(tmp_path / "log.bin")
