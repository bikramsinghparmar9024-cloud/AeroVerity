"""Back half of the pipeline (phases 8-14): verified poses -> georeferenced, trust-labelled products.

  georeferencing (GPS + gravity + barometer + optional known dimensions, DEM, satellite)
  -> per-keyframe depth, aligned to the sparse model -> cross-view agreement
  -> semantic TSDF fusion into the metric mesh
  -> bootstrap re-runs + held-out calibration -> calibrated per-vertex uncertainty
  -> trust labels -> re-fly advice
  -> GLB, PLY, LAS, splat seed, 3D Tiles, DSM/ortho/layers (GeoTIFF), GeoJSON,
     and the signed, hash-chained quality report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from aerometric.config import PipelineConfig
from aerometric.depth.crossview import CrossViewScore, score_all
from aerometric.depth.mono import AlignedDepth, DepthPrediction, align_depth, frame_observations, make_depth_model
from aerometric.fusion.tsdf import Mesh, fuse_tsdf
from aerometric.geo import LocalFrame
from aerometric.georef.dem import load_dem, register_to_dem
from aerometric.georef.gravity import gravity_from_attitude
from aerometric.georef.satellite import load_geoimage, register_to_satellite
from aerometric.georef.similarity import GeoRegistration, KnownDistance, georeference
from aerometric.outputs.mesh_io import SurfaceAttributes, write_glb, write_ply
from aerometric.outputs.points_io import write_las, write_splat_seed
from aerometric.outputs.rasters import best_view_colours, orthophoto, rasterize_surface, write_surface_rasters
from aerometric.outputs.report import generate_key, load_key, write_report
from aerometric.outputs.tiles import write_layers, write_tileset
from aerometric.pipeline.front import FrontResult
from aerometric.refly import ReflyAdvice, advise
from aerometric.trust import TrustMap, assess, vertex_agreement
from aerometric.types import DataError
from aerometric.uncertainty.bootstrap import draw_runs, pose_run, summarise, surface_offsets, surface_run
from aerometric.uncertainty.calibration import Calibration, calibrate, held_out_errors, reliability_svg


@dataclass(frozen=True)
class Anchors:
    """Optional georeferencing inputs beyond GPS/IMU/barometer."""

    known: Sequence[KnownDistance] = ()
    dem: str | Path | None = None
    satellite: str | Path | None = None


@dataclass
class BackResult:
    registration: GeoRegistration
    depths: list[AlignedDepth | None]
    scores: list[CrossViewScore | None]
    mesh: Mesh
    calibration: Calibration | None
    trust: TrustMap
    advice: list[ReflyAdvice]
    files: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class RunState:
    """What a later `compare` needs from a finished run."""

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    attrs: SurfaceAttributes
    frame: LocalFrame
    cell_m: float


def save_state(path: Path, mesh: Mesh, attrs: SurfaceAttributes, frame: LocalFrame, cell_m: float) -> Path:
    np.savez_compressed(path, vertices=mesh.vertices, faces=mesh.faces, normals=mesh.normals,
                        trust=attrs.trust, sigma=attrs.sigma_m,
                        classes=attrs.labels if attrs.labels is not None else np.full(len(mesh.vertices), 255, np.uint8),
                        frame=np.array([frame.lat0, frame.lon0, frame.alt0]), cell_m=np.array(cell_m))
    return path


def load_state(path: str | Path) -> RunState:
    d = np.load(path)
    lat0, lon0, alt0 = d["frame"]
    return RunState(d["vertices"], d["faces"], d["normals"],
                    SurfaceAttributes(d["trust"], d["sigma"], d["classes"]),
                    LocalFrame(float(lat0), float(lon0), float(alt0)), float(d["cell_m"]))


def run_back(front: FrontResult, run_dir: str | Path, cfg: PipelineConfig | None = None,
             anchors: Anchors = Anchors(), key_path: str | Path | None = None, title: str = "Flight") -> BackResult:
    cfg = cfg or PipelineConfig()
    if front.stopped or front.verification is None:
        raise DataError("the front half did not produce a verified trajectory")
    run = Path(run_dir)
    rep = front.report
    rec, rc = front.record, front.reconstruction
    traj0 = front.verification.trajectory
    t_tel = traj0.t_video_s + front.offset.offset_s
    frame = front.track.frame
    tracks, kps = rc.sparse.tracks, front.keypoints

    # 8 georeferencing ------------------------------------------------------------------------
    gravity = (gravity_from_attitude(rec.attitude, t_tel, cfg.georef.gravity_sigma_deg, cfg.georef.gimbal_roll_sign)
               if rec.attitude is not None else None)
    baro = np.interp(t_tel, rec.baro.t, rec.baro.alt_m, left=np.nan, right=np.nan) if rec.baro is not None else None
    reg = georeference(traj0, rc.sparse.points, front.gps_centers, front.gps_sigma, frame, gravity, cfg.georef,
                       baro, anchors.known)
    extra = []
    datum = "ellipsoid (GPS)"
    if anchors.dem is not None:
        d = register_to_dem(reg, load_dem(anchors.dem), cfg.georef)
        reg, datum = d.registration, "as the supplied DEM"
        extra.append(f"DEM height fit {d.height_rms_m:.2f} m")
    if anchors.satellite is not None:
        s = register_to_satellite(reg, load_geoimage(anchors.satellite), lambda k: front.images[k], cfg.georef)
        reg = s.registration
        extra.append(f"satellite matched on {s.frames_matched} keyframes")
    traj, points = reg.trajectory, reg.points
    rep.add("Georeferencing", f"Placed using {', '.join(reg.anchors)}" + (f"; {'; '.join(extra)}" if extra else "") + ".",
            {"gps_rms_m": reg.gps_rms_m, "gravity_rms_deg": reg.gravity_rms_deg, "scale_std_rel": reg.scale_std_rel,
             "tilt_constrained": reg.tilt_constrained}, reg.warnings)

    # 9 depth + cross-view agreement ---------------------------------------------------------
    model = make_depth_model(cfg.depth)
    W = traj.intrinsics.width
    preds: list[DepthPrediction | None] = []
    depths: list[AlignedDepth | None] = []
    for i, img in enumerate(front.images):
        obs = frame_observations(traj, points, tracks, kps, i)
        w = min(cfg.depth.work_width, W)
        h = round(img.shape[0] * w / W)
        small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if w < W else img
        p = model.predict(small, obs, W)
        preds.append(p)
        depths.append(align_depth(p, obs, W, cfg.depth))
    scores = score_all(traj, depths, cfg.depth)
    good = [d for d in depths if d is not None and d.good]
    agree = [s.agreeing_fraction() for s in scores if s is not None]
    rep.add("Depth", f"{len(good)} of {len(depths)} keyframes have usable depth maps ({model.name}).",
            {"median_rel_error": float(np.median([d.median_rel_error for d in good])) if good else None,
             "median_agreeing_fraction": float(np.median(agree)) if agree else None},
            ["depth maps come from the stand-in (interpolated sparse depth), not a depth network"]
            if model.is_mock else [])

    # 10 semantic fusion -----------------------------------------------------------------------
    mesh = fuse_tsdf(traj, depths, scores, front.segmentations, cfg.fusion, cfg.semantic)
    rep.add("Surface", f"Mesh with {len(mesh.vertices):,} vertices and {len(mesh.faces):,} faces "
                       f"({mesh.voxel_m:.2f} m voxels).",
            {"classes": mesh.semantics.class_counts() if mesh.semantics else {}}, mesh.warnings)

    # 11 uncertainty: bootstrap + calibration ---------------------------------------------------
    draws = draw_runs(len(traj), front.gps_centers, front.gps_sigma, cfg.uncertainty, cfg.seed)
    # re-runs only measure how much the surface moves, so they use fewer points and coarser voxels
    boot_classical = replace(cfg.classical, max_points=cfg.uncertainty.bootstrap_max_points)
    boot_fusion = replace(cfg.fusion, voxel_m=mesh.voxel_m * cfg.uncertainty.bootstrap_voxel_factor)
    run_pts, offsets, kept = [], [], []
    for dr in draws:
        try:
            pr = pose_run(dr, kps, front.pairs, traj, front.gps_sigma, tracks, boot_classical)
            verts = surface_run(pr, traj, preds, tracks, kps, W, cfg.depth, boot_fusion)
        except DataError:
            continue  # a subset too sparse to rebuild simply contributes nothing
        kept.append(dr)
        run_pts.append(pr.points)
        offsets.append(surface_offsets(mesh.vertices, mesh.normals, verts, cfg.uncertainty.max_gap_m))
    cal = None
    if offsets:
        surf = summarise(np.stack(offsets), cfg.uncertainty)
        pt = summarise(np.stack(run_pts), cfg.uncertainty)
        ho = held_out_errors(run_pts, kept, traj, tracks, kps)
        if len(ho.error_m):
            cal = calibrate(pt.std[ho.point_ids], ho.error_m, cfg.uncertainty)
        sigma = surf.std * (cal.scale if cal is not None else 1.0)
    else:
        sigma = np.full(len(mesh.vertices), np.nan)
    rep.add("Uncertainty", f"{len(offsets)} of {len(draws)} bootstrap runs completed"
                           + (f"; uncertainty scaled by {cal.scale:.2f} after checking held-out views." if cal else "."),
            {"median_sigma_m": float(np.nanmedian(sigma)) if np.isfinite(sigma).any() else None,
             "calibration_samples": cal.n if cal else 0}, cal.warnings if cal else ["uncertainty could not be calibrated"])

    # 12 trust ------------------------------------------------------------------------------
    agreement = vertex_agreement(mesh.vertices, traj, depths, scores, cfg.trust)
    sem = mesh.semantics
    mock = model.is_mock or any(s.is_mock for s in front.segmentations)
    trust = assess(sigma, agreement, mesh.vertex_weight, sem.dynamic if sem else None, mock, cfg.trust)
    fr = trust.fractions(mesh.vertices, mesh.faces)
    rep.add("Trust", f"{fr['measured']:.0%} of the surface is Measured, {fr['inferred']:.0%} Inferred, "
                     f"{fr['unknown']:.0%} Unknown.", fr, trust.warnings)

    # 13 re-fly advice --------------------------------------------------------------------
    gaps = front.coverage.regions if front.coverage is not None else ()
    advice = advise(mesh.vertices, mesh.faces, mesh.normals, trust, traj, sem.dynamic if sem else None, gaps,
                    frame=frame, cfg=cfg.refly)
    out = run / "products"
    out.mkdir(parents=True, exist_ok=True)
    adv_path = out / "refly_advice.json"
    adv_path.write_text(json.dumps([{"message": a.message, "reason": a.reason, "area_m2": a.area_m2,
                                     "viewpoints": [{"lat": v.lat, "lon": v.lon, "alt_m": v.alt_m,
                                                     "heading_deg": v.heading_deg, "gimbal_pitch_deg": v.gimbal_pitch_deg}
                                                    for v in a.viewpoints]} for a in advice], indent=1),
                        encoding="utf-8")
    rep.add("Re-fly advice", advice[0].message if advice else "No weak areas large enough to need another pass.",
            {"regions": len(advice)}, outputs=[adv_path], root=run)

    # 14 outputs --------------------------------------------------------------------------
    attrs = SurfaceAttributes(trust.labels, trust.effective_sigma_m, sem.labels if sem else None)
    files = {"refly": adv_path,
             "glb": write_glb(out / "model.glb", mesh.vertices, mesh.faces, mesh.normals, attrs, frame, title=title),
             "mesh_ply": write_ply(out / "model.ply", mesh.vertices, mesh.faces, mesh.normals, attrs, frame),
             "sparse_ply": write_ply(out / "sparse.ply", points, frame=frame)}
    rgb, _ = best_view_colours(mesh.vertices, traj, lambda k: front.images[k], depths, cfg.raster.occlusion_tol)
    files["las"] = write_las(out / "points.las", mesh.vertices, frame, rgb, attrs.labels, attrs.trust)
    files["splat"] = write_splat_seed(out / "splat_seed.ply", mesh.vertices, mesh.normals, rgb, attrs.trust, frame)
    files["tileset"] = write_tileset(out / "tiles", mesh.vertices, mesh.faces, mesh.normals, attrs, frame,
                                     height_datum=datum)
    ras = rasterize_surface(mesh.vertices, mesh.faces, attrs, cfg.raster)
    ortho, ow = orthophoto(ras, traj, lambda k: front.images[k], depths, cfg.raster)
    files.update(write_surface_rasters(out, ras, frame, ortho, cfg.raster))
    files.update({f"{k}_geojson": v for k, v in write_layers(out, ras.classes, ras.trust, ras.grid, frame).items()})
    files["state"] = save_state(out / "state.npz", mesh, attrs, frame, ras.grid.cell_m)
    product_files = [p for k, p in files.items() if k != "tileset"] + list((out / "tiles").glob("*"))
    rep.add("Outputs", f"Wrote {len(product_files)} files (mesh, point clouds, tiles, rasters, layers).",
            {"raster_cell_m": ras.grid.cell_m}, ow, outputs=product_files, root=run)

    key = (load_key(key_path) if key_path is not None and Path(key_path).is_file()
           else generate_key(key_path or run / "signing_key.pem"))
    overview = {"trust": fr, "gps_rms_m": reg.gps_rms_m, "anchors": ", ".join(reg.anchors),
                "keyframes": len(traj), "mesh_vertices": len(mesh.vertices),
                "uncertainty_scale": cal.scale if cal else None,
                "verdict": "Completed with notes" if any(e["warnings"] for e in rep.entries) else "Completed"}
    doc = rep.finalize(key, overview)
    js, page = write_report(run / "report", doc, reliability_svg(cal) if cal else None)
    files.update({"report_json": js, "report_html": page})
    return BackResult(reg, depths, scores, mesh, cal, trust, advice, files)
