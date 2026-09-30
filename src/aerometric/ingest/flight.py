"""Assemble all ingestion inputs into one FlightRecord."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aerometric.config import PipelineConfig
from aerometric.ingest.intrinsics import load_intrinsics
from aerometric.ingest.rtk import parse_rtklib_pos
from aerometric.ingest.telemetry import load_telemetry, parse_baro_csv, parse_imu_csv
from aerometric.ingest.video import probe_video
from aerometric.types import CameraIntrinsics, FlightRecord

DURATION_MISMATCH = 0.10  # warn when telemetry and video lengths differ by more than 10 %


@dataclass(frozen=True)
class FlightInputs:
    video: str | Path
    telemetry: str | Path
    imu: str | Path | None = None
    baro: str | Path | None = None
    rtk: str | Path | None = None
    intrinsics: str | Path | None = None


def ingest_flight(inputs: FlightInputs, cfg: PipelineConfig | None = None) -> FlightRecord:
    cfg = cfg or PipelineConfig()
    warnings: list[str] = []

    video = probe_video(inputs.video, cfg.paths)
    tele = load_telemetry(inputs.telemetry, cfg.telemetry)
    warnings += tele.warnings

    span = float(tele.gps.t[-1] - tele.gps.t[0])
    if abs(span - video.duration_s) > DURATION_MISMATCH * video.duration_s:
        warnings.append(f"telemetry covers {span:.1f}s but video is {video.duration_s:.1f}s; "
                        "check that the files belong to the same flight")

    if inputs.intrinsics is not None:
        intrinsics = load_intrinsics(inputs.intrinsics, video.width, video.height)
    else:
        intrinsics = CameraIntrinsics.from_fov(video.width, video.height, cfg.ingest.default_hfov_deg)
        warnings.append(f"no camera intrinsics supplied; assuming {cfg.ingest.default_hfov_deg:g}° "
                        "horizontal FOV until self-calibration")

    imu = parse_imu_csv(inputs.imu, cfg.telemetry) if inputs.imu is not None else None
    baro = parse_baro_csv(inputs.baro, cfg.telemetry) if inputs.baro is not None else tele.baro

    rtk = None
    if inputs.rtk is not None:
        res = parse_rtklib_pos(inputs.rtk, cfg.ingest.rtk_source,  # type: ignore[arg-type]
                               tuple(cfg.ingest.rtk_accept_quality))
        rtk = res.track
        warnings += res.warnings
        if res.fix_ratio < 0.5:
            warnings.append(f"RTK/PPK fixed solution on only {res.fix_ratio:.0%} of epochs")

    return FlightRecord(video=video, gps=tele.gps, intrinsics=intrinsics, attitude=tele.attitude,
                        imu=imu, baro=baro, rtk=rtk, warnings=tuple(warnings))
