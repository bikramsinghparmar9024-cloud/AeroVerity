"""Command line: run a flight, compare two flights, verify a report, open the viewer.

  python -m dronerecon run --video V.MP4 --telemetry V.SRT --out runs/flight1 [--viewer]
  python -m dronerecon compare runs/flight1 runs/flight2 --out runs/change
  python -m dronerecon verify runs/flight1/report/quality_report.json --files runs/flight1
  python -m dronerecon viewer runs/flight1
  python -m dronerecon app --runs runs        (upload flights in the browser)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dronerecon.config import ConfigError, load_config
from dronerecon.ingest.flight import FlightInputs
from dronerecon.outputs.report import ReportBuilder, verify_report
from dronerecon.types import DataError


class ConsoleReport(ReportBuilder):
    """Report builder that also tells the operator what just happened."""

    def add(self, stage, summary, *args, **kwargs):
        entry = super().add(stage, summary, *args, **kwargs)
        print(f"  {stage:<20} {summary}")
        for w in entry["warnings"]:
            print(f"  {'':<20} note: {w[0].upper() + w[1:]}")
        return entry


def _cmd_run(args) -> int:
    from dronerecon.pipeline.back import Anchors, run_back
    from dronerecon.pipeline.front import run_front

    cfg = load_config(args.config)
    out = Path(args.out)
    title = args.title or Path(args.video).stem
    if args.viewer:
        from dronerecon.viewer.server import serve_in_background

        _, url = serve_in_background(out, port=args.port)
        print(f"Viewer: {url}  (updates as the run progresses)")
    rep = ConsoleReport(title, out.name, cfg.to_dict())
    inputs = FlightInputs(args.video, args.telemetry, args.imu, args.baro, args.rtk, args.intrinsics)
    t0 = time.time()
    print(f"Reconstructing {title}")
    front = run_front(inputs, out, cfg, rep, title, force=args.force)
    if front.stopped:
        print("\nStopped: the pre-flight check does not expect a usable model from this flight.")
        print("Run again with --force to try anyway.")
        return 2
    back = run_back(front, out, cfg, Anchors(dem=args.dem, satellite=args.satellite), args.key, title)
    fr = back.trust.fractions(back.mesh.vertices, back.mesh.faces)
    print(f"\nDone in {time.time() - t0:.0f} s. {fr['measured']:.0%} of the surface is Measured, "
          f"{fr['inferred']:.0%} Inferred, {fr['unknown']:.0%} Unknown.")
    print(f"Report:   {back.files['report_html']}")
    print(f"Products: {out / 'products'}")
    if args.viewer:
        print("Viewer is still running; press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


def _cmd_compare(args) -> int:
    from dronerecon.pipeline.compare import compare_runs

    cfg = load_config(args.config)
    out = Path(args.out)
    print(f"Comparing {args.run_a} (A) with {args.run_b} (B)")
    res = compare_runs(args.run_a, args.run_b, out, cfg, args.key,
                       ConsoleReport("Change between flights", out.name, cfg.to_dict()))
    print(f"\nReport: {res.files['report_html']}")
    return 0


def _cmd_verify(args) -> int:
    doc = json.loads(Path(args.report).read_text(encoding="utf-8"))
    check = verify_report(doc, files_root=args.files, trusted_public_key=args.public_key)
    if check.ok:
        extra = " and every output file matches" if check.files_ok else ""
        print(f"Report is intact: the chain and signature are valid{extra}.")
        return 0
    print("Report did NOT verify:")
    for p in check.problems:
        print(f"  - {p[0].upper() + p[1:]}")
    return 1


def _cmd_viewer(args) -> int:
    from dronerecon.viewer.server import make_server

    server = make_server(args.run_dir, port=args.port)
    print(f"Viewer running at http://127.0.0.1:{server.server_address[1]}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def _cmd_app(args) -> int:
    from dronerecon.viewer.jobs import JobQueue
    from dronerecon.viewer.server import make_app_server

    jobs = JobQueue(Path(args.runs), load_config(args.config), Path(args.key) if args.key else None)
    server = make_app_server(jobs, port=args.port)
    print(f"dronerecon is running at http://127.0.0.1:{server.server_address[1]}/")
    print(f"Runs are stored in {Path(args.runs).resolve()}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dronerecon", description="Single-pass drone video to a georeferenced 3D model.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="reconstruct one flight")
    r.add_argument("--video", required=True, help="drone video (MP4/MOV)")
    r.add_argument("--telemetry", required=True, help="DJI .SRT sidecar or flight-log .csv")
    r.add_argument("--out", required=True, help="output folder for this run")
    r.add_argument("--imu", help="IMU .csv (optional)")
    r.add_argument("--baro", help="barometer .csv (optional)")
    r.add_argument("--rtk", help="RTKLIB .pos from RTK/PPK (optional)")
    r.add_argument("--intrinsics", help="camera calibration .json (optional)")
    r.add_argument("--dem", help="elevation model .asc/.tif in WGS84 (optional)")
    r.add_argument("--satellite", help="georeferenced image .tif or .png+.pgw (optional)")
    r.add_argument("--config", help="settings .json (defaults otherwise)")
    r.add_argument("--key", help="signing key .pem (created if missing)")
    r.add_argument("--title", help="name shown in the viewer and report")
    r.add_argument("--force", action="store_true", help="continue even if the pre-flight check rejects the flight")
    r.add_argument("--viewer", action="store_true", help="open the live viewer while running")
    r.add_argument("--port", type=int, default=8765)
    r.set_defaults(func=_cmd_run)

    c = sub.add_parser("compare", help="map changes between two finished runs")
    c.add_argument("run_a", help="earlier run folder (reference)")
    c.add_argument("run_b", help="later run folder")
    c.add_argument("--out", required=True)
    c.add_argument("--config")
    c.add_argument("--key")
    c.set_defaults(func=_cmd_compare)

    v = sub.add_parser("verify", help="check a quality report has not been altered")
    v.add_argument("report", help="quality_report.json")
    v.add_argument("--files", help="run folder, to also check every output file")
    v.add_argument("--public-key", help="only accept reports signed with this base64 key")
    v.set_defaults(func=_cmd_verify)

    a = sub.add_parser("app", help="local web app: upload flights, follow progress, download results")
    a.add_argument("--runs", default="runs", help="folder where uploaded runs are stored")
    a.add_argument("--config", help="settings .json used for every run")
    a.add_argument("--key", help="signing key .pem shared by all runs (default: one per run)")
    a.add_argument("--port", type=int, default=8765)
    a.set_defaults(func=_cmd_app)

    w = sub.add_parser("viewer", help="open the viewer for a run folder")
    w.add_argument("run_dir")
    w.add_argument("--port", type=int, default=8765)
    w.set_defaults(func=_cmd_viewer)
    return p


def _safe_console() -> None:
    """Never let a character the console cannot show (σ, °, m²) crash a run.

    Redirected output on Windows uses the legacy code page (cp1252), which lacks
    these; unencodable characters are replaced instead of raising.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # not a text stream we can reconfigure (e.g. captured)
            pass


def main(argv: list[str] | None = None) -> int:
    _safe_console()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (DataError, ConfigError, FileNotFoundError) as exc:
        print(f"\nCould not continue: {exc}", file=sys.stderr)
        return 1
