"""Flight telemetry parsers: DJI SRT sidecars and generic CSV logs (GPS, IMU, baro).

All returned times are seconds on the telemetry clock. For SRT files that is
the subtitle start time, which is nominally the video clock but routinely
drifts or lags; the sync stage estimates the real offset.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerometric.config import TelemetryConfig
from aerometric.types import AttitudeSeries, BaroSeries, DataError, GpsTrack, ImuSeries

FT_TO_M = 0.3048


@dataclass(frozen=True)
class TelemetryResult:
    gps: GpsTrack
    attitude: AttitudeSeries | None = None
    baro: BaroSeries | None = None  # relative (take-off) altitude
    warnings: tuple[str, ...] = ()


# --------------------------------------------------------------------------- helpers

def _clean_series(t: np.ndarray, *cols: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], int]:
    """Sort by time and drop non-finite times and duplicate timestamps (keep first)."""
    keep = np.isfinite(t)
    t, cols = t[keep], [c[keep] for c in cols]
    order = np.argsort(t, kind="stable")
    t, cols = t[order], [c[order] for c in cols]
    _, first = np.unique(t, return_index=True)
    dropped = len(t) - len(first)
    return t[first], [c[first] for c in cols], dropped


def _build_track(t, lat, lon, alt, drop_no_fix: bool, extra: dict[str, np.ndarray],
                 warnings: list[str], name: str) -> tuple[GpsTrack, dict[str, np.ndarray]]:
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt)
    if drop_no_fix:
        ok &= ~((lat == 0) & (lon == 0))
    if (bad := int((~ok).sum())):
        warnings.append(f"{name}: dropped {bad} sample(s) without a valid GPS fix")
    keys = list(extra)
    t, cols, dup = _clean_series(t[ok], lat[ok], lon[ok], alt[ok], *(extra[k][ok] for k in keys))
    if dup:
        warnings.append(f"{name}: dropped {dup} sample(s) with duplicate timestamps")
    if len(t) < 2:
        raise DataError(f"{name}: fewer than 2 usable GPS samples")
    return GpsTrack(t, cols[0], cols[1], cols[2]), dict(zip(keys, cols[3:]))


# --------------------------------------------------------------------------- DJI SRT

_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->")
_BRACKET = re.compile(r"\[([^\]]*)\]")
_KEY_VALUE = re.compile(r"([a-z_]+)\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.I)
_LEGACY_GPS = re.compile(r"GPS\s*\(\s*([-+\d.]+)\s*,\s*([-+\d.]+)\s*,\s*([-+\d.]+)", re.I)
_LEGACY_BARO = re.compile(r"BAROMETER\s*:\s*([-+\d.]+)", re.I)
_TAGS = re.compile(r"<[^>]+>")

_KEY_ALIASES = {"longtitude": "longitude", "lat": "latitude", "lon": "longitude",
                "altitude": "abs_alt", "gb_yaw": "yaw", "gb_pitch": "pitch", "gb_roll": "roll"}


def _parse_srt_block(block: str) -> dict[str, float] | None:
    m = _SRT_TIME.search(block)
    if not m:
        return None
    h, mi, s, frac = m.groups()
    rec: dict[str, float] = {"t": int(h) * 3600 + int(mi) * 60 + int(s) + int(frac) / 10 ** len(frac)}
    body = _TAGS.sub(" ", block[m.end():])
    for group in _BRACKET.findall(body):  # a bracket may hold several "key: value" pairs
        for key, val in _KEY_VALUE.findall(group):
            rec[_KEY_ALIASES.get(key.lower(), key.lower())] = float(val)
    if (lg := _LEGACY_GPS.search(body)):  # older firmware: GPS(lon, lat, alt)
        rec["longitude"], rec["latitude"] = float(lg.group(1)), float(lg.group(2))
        rec.setdefault("abs_alt", float(lg.group(3)))
    if (lb := _LEGACY_BARO.search(body)):
        rec["rel_alt"] = float(lb.group(1))
    return rec


def parse_dji_srt(path: str | Path, cfg: TelemetryConfig | None = None) -> TelemetryResult:
    cfg = cfg or TelemetryConfig()
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    recs = [r for b in re.split(r"\n\s*\n", text) if (r := _parse_srt_block(b))]
    recs = [r for r in recs if "latitude" in r and "longitude" in r]
    if not recs:
        raise DataError(f"{p.name}: no GPS fields found in SRT")

    def col(key: str) -> np.ndarray:
        return np.array([r.get(key, np.nan) for r in recs], dtype=np.float64)

    warnings: list[str] = []
    alt = col("abs_alt")
    if np.all(np.isnan(alt)):
        alt = col("rel_alt")
        warnings.append(f"{p.name}: no absolute altitude; using relative altitude as GPS altitude")
    extra = {k: col(k) for k in ("rel_alt", "yaw", "pitch", "roll")}
    track, ex = _build_track(col("t"), col("latitude"), col("longitude"), alt,
                             cfg.drop_no_fix, extra, warnings, p.name)

    baro = BaroSeries(track.t, ex["rel_alt"]) if np.all(np.isfinite(ex["rel_alt"])) else None
    ypr = np.column_stack([ex["yaw"], ex["pitch"], ex["roll"]])
    attitude = AttitudeSeries(track.t, ypr) if np.all(np.isfinite(ypr)) else None
    return TelemetryResult(track, attitude, baro, tuple(warnings))


# --------------------------------------------------------------------------- CSV

_TIME_ALIASES = ("time_s", "t", "time", "timestamp", "time(millisecond)", "time_ms",
                 "offsettime", "gps_time")
_LAT_ALIASES = ("latitude", "lat", "osd.latitude")
_LON_ALIASES = ("longitude", "lon", "lng", "osd.longitude")
_ALT_ALIASES = ("alt_m", "altitude", "abs_alt", "altitude_above_sealevel(feet)",
                "altitude_above_sealevel(meters)", "alt", "osd.altitude [m]")


def read_csv_columns(path: str | Path) -> dict[str, np.ndarray]:
    """Read a CSV into float columns keyed by lower-cased header. Non-numeric cells become NaN."""
    p = Path(path)
    with p.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise DataError(f"{p.name}: CSV has no data rows")
    header = [h.strip().lower() for h in rows[0]]

    def num(v: str) -> float:
        try:
            return float(v)
        except ValueError:
            return np.nan

    data = np.array([[num(v) for v in r[: len(header)]] + [np.nan] * (len(header) - len(r))
                     for r in rows[1:] if any(c.strip() for c in r)], dtype=np.float64)
    return {h: data[:, i] for i, h in enumerate(header) if h}


def _pick(cols: dict[str, np.ndarray], override: str | None, aliases: tuple[str, ...],
          what: str, source: str, required: bool = True) -> tuple[str, np.ndarray] | None:
    names = (override.lower(),) if override else aliases
    for n in names:
        if n in cols:
            return n, cols[n]
    if required:
        raise DataError(f"{source}: no {what} column (looked for {', '.join(names)})")
    return None


def _time_seconds(name: str, t: np.ndarray) -> np.ndarray:
    if "millisecond" in name or name.endswith("_ms"):
        t = t / 1000.0
    return t - t[np.isfinite(t)].min()


def parse_gps_csv(path: str | Path, cfg: TelemetryConfig | None = None) -> TelemetryResult:
    cfg = cfg or TelemetryConfig()
    p = Path(path)
    cols = read_csv_columns(p)
    try:
        tn, t = _pick(cols, cfg.time_col, _TIME_ALIASES, "time", p.name)
        _, lat = _pick(cols, cfg.lat_col, _LAT_ALIASES, "latitude", p.name)
        _, lon = _pick(cols, cfg.lon_col, _LON_ALIASES, "longitude", p.name)
        an, alt = _pick(cols, cfg.alt_col, _ALT_ALIASES, "altitude", p.name)
    except DataError as exc:
        found = ", ".join(list(cols)[:8])
        raise DataError(f"This CSV does not look like a flight log ({exc}; its columns are: {found}). "
                        "For a DJI flight, choose the .SRT file with the same name as the video; a flight-log "
                        "CSV needs time, latitude, longitude and altitude columns.") from None
    if "feet" in an or "(ft)" in an:
        alt = alt * FT_TO_M
    warnings: list[str] = []
    track, _ = _build_track(_time_seconds(tn, t), lat, lon, alt, cfg.drop_no_fix, {}, warnings, p.name)
    return TelemetryResult(track, warnings=tuple(warnings))


def parse_imu_csv(path: str | Path, cfg: TelemetryConfig | None = None) -> ImuSeries:
    """Columns: time + gx, gy, gz (gyro) and optionally ax, ay, az (accel, m/s^2)."""
    cfg = cfg or TelemetryConfig()
    p = Path(path)
    cols = read_csv_columns(p)
    tn, t = _pick(cols, cfg.time_col, _TIME_ALIASES, "time", p.name)
    gyro = np.column_stack([_pick(cols, None, (f"g{a}", f"gyro_{a}"), f"gyro {a}", p.name)[1]
                            for a in "xyz"])
    if cfg.gyro_unit == "deg/s":
        gyro = np.radians(gyro)
    acc_cols = [_pick(cols, None, (f"a{a}", f"acc_{a}", f"accel_{a}"), "accel", p.name, False)
                for a in "xyz"]
    accel = np.column_stack([c[1] for c in acc_cols]) if all(acc_cols) else None
    arrays = [gyro] + ([accel] if accel is not None else [])
    ts, cleaned, _ = _clean_series(_time_seconds(tn, t), *arrays)
    ok = np.all(np.isfinite(cleaned[0]), axis=1)
    return ImuSeries(ts[ok], cleaned[0][ok], cleaned[1][ok] if accel is not None else None)


def parse_baro_csv(path: str | Path, cfg: TelemetryConfig | None = None) -> BaroSeries:
    """Columns: time + baro altitude in metres (alt_m / baro_alt / rel_alt / pressure_alt)."""
    cfg = cfg or TelemetryConfig()
    p = Path(path)
    cols = read_csv_columns(p)
    tn, t = _pick(cols, cfg.time_col, _TIME_ALIASES, "time", p.name)
    _, alt = _pick(cols, None, ("baro_alt", "rel_alt", "pressure_alt", "alt_m"), "baro altitude", p.name)
    ts, (a,), _ = _clean_series(_time_seconds(tn, t), alt)
    ok = np.isfinite(a)
    return BaroSeries(ts[ok], a[ok])


def load_telemetry(path: str | Path, cfg: TelemetryConfig | None = None) -> TelemetryResult:
    """Dispatch on file extension (.srt or .csv)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".srt":
        return parse_dji_srt(path, cfg)
    if suffix == ".csv":
        return parse_gps_csv(path, cfg)
    raise DataError(f"unsupported telemetry format '{suffix}' (expected .srt or .csv)")
