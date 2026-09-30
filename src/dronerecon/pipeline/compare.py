"""Compare two finished runs (phase 15): register B onto A, then map what changed."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dronerecon.change.changemap import ChangeMap, compute_change, describe, write_change
from dronerecon.change.register import FlightRegistration, apply, register_flights
from dronerecon.config import PipelineConfig, RasterConfig
from dronerecon.outputs.rasters import make_grid, rasterize_surface
from dronerecon.outputs.report import ReportBuilder, generate_key, load_key, write_report
from dronerecon.pipeline.back import load_state


@dataclass
class CompareResult:
    registration: FlightRegistration
    change: ChangeMap
    files: dict[str, Path]


def compare_runs(run_a: str | Path, run_b: str | Path, out_dir: str | Path, cfg: PipelineConfig | None = None,
                 key_path: str | Path | None = None, report: ReportBuilder | None = None) -> CompareResult:
    cfg = cfg or PipelineConfig()
    a = load_state(Path(run_a) / "products" / "state.npz")
    b = load_state(Path(run_b) / "products" / "state.npz")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = report or ReportBuilder("Change between flights", out.name, cfg.to_dict())

    reg = register_flights(a.vertices, a.frame, b.vertices, b.frame, a.normals, cfg=cfg.change)
    rep.add("Flight registration", f"Flight B aligned to flight A to {reg.rms_m:.2f} m "
                                   f"({reg.overlap:.0%} overlap, {reg.iterations} iterations).",
            {"rms_m": reg.rms_m, "overlap": reg.overlap, "converged": reg.converged}, reg.warnings)

    rcfg = RasterConfig(cell_m=a.cell_m, fill_holes_px=cfg.raster.fill_holes_px, nodata=cfg.raster.nodata)
    grid = make_grid(a.vertices, a.faces, rcfg)
    ra = rasterize_surface(a.vertices, a.faces, a.attrs, rcfg, grid)
    rb = rasterize_surface(apply(reg, b.vertices, b.frame, a.frame), b.faces, b.attrs, rcfg, grid)
    cm = compute_change(ra, rb, reg.rms_m, cfg.change)
    files = write_change(out, cm, ra, a.frame)
    summary = out / "change_summary.json"
    summary.write_text(json.dumps({**cm.summary, "lines": describe(cm)}, indent=1), encoding="utf-8")
    files["summary"] = summary
    rep.add("Change map", " ".join(describe(cm)[:2]), cm.summary, outputs=list(files.values()), root=out)

    key = (load_key(key_path) if key_path is not None and Path(key_path).is_file()
           else generate_key(key_path or out / "signing_key.pem"))
    doc = rep.finalize(key, {**cm.summary, "verdict": "Completed with notes" if reg.warnings else "Completed"})
    js, page = write_report(out / "report", doc)
    files.update({"report_json": js, "report_html": page})
    return CompareResult(reg, cm, files)
