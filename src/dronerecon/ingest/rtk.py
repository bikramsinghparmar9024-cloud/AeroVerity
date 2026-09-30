"""RTKLIB `.pos` solution files (RTKPOST / RTKNAVI output).

Times are GPS time. The returned track uses seconds since the first accepted
epoch; the absolute GPST start is returned separately so the sync stage can
place it on the telemetry clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dronerecon.types import DataError, GpsSource, GpsTrack

_GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
_WEEK_S = 7 * 86400


@dataclass(frozen=True)
class RtkResult:
    track: GpsTrack
    start_gpst_s: float  # seconds since the GPS epoch of track.t == 0
    fix_ratio: float  # share of accepted epochs with Q == 1
    warnings: tuple[str, ...] = ()


def _epoch_seconds(tokens: list[str]) -> tuple[float, int]:
    """Return (GPST seconds since GPS epoch, tokens consumed)."""
    if "/" in tokens[0]:
        stamp = datetime.strptime(f"{tokens[0]} {tokens[1]}", "%Y/%m/%d %H:%M:%S.%f")
        return (stamp.replace(tzinfo=timezone.utc) - _GPS_EPOCH).total_seconds(), 2
    return int(tokens[0]) * _WEEK_S + float(tokens[1]), 2  # GPS week + time of week


def parse_rtklib_pos(path: str | Path, source: GpsSource = "ppk",
                     accept_quality: tuple[int, ...] = (1, 2)) -> RtkResult:
    if source == "gnss":
        raise DataError("RTK track source must be 'rtk' or 'ppk'")
    p = Path(path)
    rows: list[list[float]] = []
    skipped = 0
    for line_no, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip() or line.startswith("%"):
            continue
        tok = line.split()
        try:
            t, used = _epoch_seconds(tok)
            lat, lon, h, q = (float(v) for v in tok[used:used + 4])
            sdn, sde, sdu = (float(v) for v in tok[used + 5:used + 8])
        except (ValueError, IndexError):
            raise DataError(f"{p.name}:{line_no}: unrecognised .pos line (expected lat/lon/height "
                            "in degrees; ECEF or DMS output is not supported)") from None
        if int(q) in accept_quality:
            rows.append([t, lat, lon, h, q, sdn, sde, sdu])
        else:
            skipped += 1
    if len(rows) < 2:
        raise DataError(f"{p.name}: fewer than 2 epochs with quality in {accept_quality}")

    a = np.array(rows)
    a = a[np.unique(a[:, 0], return_index=True)[1]]
    warnings = [f"{p.name}: skipped {skipped} epoch(s) below the accepted quality"] if skipped else []
    track = GpsTrack(
        t=a[:, 0] - a[0, 0], lat=a[:, 1], lon=a[:, 2], alt_m=a[:, 3], source=source,
        h_acc_m=np.hypot(a[:, 5], a[:, 6]), v_acc_m=a[:, 7],
    )
    return RtkResult(track, float(a[0, 0]), float(np.mean(a[:, 4] == 1)), tuple(warnings))
