"""Front half of the pipeline (phases 1-7): video + telemetry -> verified trajectory and sparse model.

Stages, in architecture order:
  ingest -> time offset -> GPS smoothing -> pre-flight gate -> keyframes ->
  segmentation -> features -> pair selection -> matching -> self-calibration ->
  AI pose (T0 preview) -> classical GPS-constrained pose -> pose verifier ->
  first sparse model + coverage check (fed back to the gate).
Viewer files are written as soon as each is available, and every stage adds an
entry to the quality report.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from aerometric.calibration import PairCorrespondences, self_calibrate
from aerometric.config import PipelineConfig
from aerometric.ingest.flight import FlightInputs, ingest_flight
from aerometric.ingest.video import iter_frames
from aerometric.matching.features import Features, make_matching, verify_pair
from aerometric.matching.pairs import global_descriptors, select_pairs
from aerometric.outputs.report import ReportBuilder
from aerometric.pose.classical import ReconstructionResult, reconstruct
from aerometric.pose.feedforward import PoseInputs, make_pose_model, t0_preview
from aerometric.georef.gravity import rotation_from_gimbal
from aerometric.pose.tracks import PairMatches
from aerometric.pose.types import CameraPose, Trajectory
from aerometric.pose.verifier import VerificationResult, verify
from aerometric.quality.coverage import CoverageReport, analyse_coverage, with_coverage
from aerometric.quality.gate import GateReport, evaluate_gate, frame_metrics, sample_stride
from aerometric.quality.keyframes import KeyframeSet, select_keyframes
from aerometric.segmentation import SegmentationResult, make_segmenter
from aerometric.sync.offset import OffsetEstimate, estimate_rtk_offset, estimate_video_offset, flow_signal
from aerometric.sync.smoothing import SmoothedTrack, smooth_flight
from aerometric.types import DataError, FlightRecord
from aerometric.viewer.export import write_accepted, write_preview


@dataclass
class FrontResult:
    record: FlightRecord
    offset: OffsetEstimate
    track: SmoothedTrack
    gate: GateReport
    report: ReportBuilder
    stopped: bool = False  # True when the gate rejected the flight
    keyframes: KeyframeSet | None = None
    images: list[np.ndarray] = field(default_factory=list)  # BGR keyframe images
    segmentations: list[SegmentationResult] = field(default_factory=list)
    features: list[Features] = field(default_factory=list)
    pairs: list[PairMatches] = field(default_factory=list)
    gps_centers: np.ndarray | None = None
    gps_sigma: np.ndarray | None = None
    reconstruction: ReconstructionResult | None = None
    verification: VerificationResult | None = None
    coverage: CoverageReport | None = None

    @property
    def keypoints(self) -> list[np.ndarray]:
        return [f.keypoints.astype(float) for f in self.features]


def _classical_start(record: FlightRecord, preview: Trajectory, t_tel: np.ndarray, gps: np.ndarray,
                     cfg: PipelineConfig) -> Trajectory:
    """Starting poses for bundle adjustment.

    With gimbal attitude in the telemetry: GPS positions + gimbal orientation, which does
    not drift. Otherwise the T0 preview. Only the starting point differs; the adjustment
    itself is the same GPS-constrained bundle adjustment either way.
    """
    att = record.attitude
    if att is None or t_tel[0] < att.t[0] - 1.0 or t_tel[-1] > att.t[-1] + 1.0:
        return preview
    yaw = np.interp(t_tel, att.t, np.degrees(np.unwrap(np.radians(att.ypr_deg[:, 0]))))
    pitch = np.interp(t_tel, att.t, att.ypr_deg[:, 1])
    roll = np.interp(t_tel, att.t, att.ypr_deg[:, 2])
    poses = tuple(CameraPose.from_center(rotation_from_gimbal(y, p, r, cfg.georef.gimbal_roll_sign), c)
                  for y, p, r, c in zip(yaw, pitch, roll, gps))
    return replace(preview, poses=poses, valid=np.ones(len(poses), bool), source="gps+gimbal", is_mock=False,
                   warnings=())


def run_front(inputs: FlightInputs, run_dir: str | Path, cfg: PipelineConfig | None = None,
              report: ReportBuilder | None = None, title: str = "Flight", force: bool = False) -> FrontResult:
    """force: continue even when the pre-flight gate rejects the flight."""
    cfg = cfg or PipelineConfig()
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    rep = report or ReportBuilder(title, run.name, cfg.to_dict())

    # 1 ingest -------------------------------------------------------------------------------
    record = ingest_flight(inputs, cfg)
    v = record.video
    rep.add("Ingestion", f"{v.width}x{v.height} video, {v.frame_count} frames at {v.fps:.2f} fps; "
                         f"{len(record.gps)} GPS fixes.",
            {"duration_s": v.duration_s, "intrinsics": record.intrinsics.source,
             "imu": record.imu is not None, "rtk": record.rtk is not None},
            record.warnings, inputs=[p for p in (inputs.video, inputs.telemetry, inputs.imu, inputs.baro,
                                                inputs.rtk, inputs.intrinsics) if p is not None])

    # 2 sync + smoothing ---------------------------------------------------------------------
    stride = max(1, round(v.fps / 10))  # ~10 fps is plenty for motion signals
    flow = flow_signal(iter_frames(v, stride=stride), cfg.sync)
    offset = estimate_video_offset(flow, record.gps, record.imu, cfg.sync)
    rtk_off = estimate_rtk_offset(record.gps, record.rtk, cfg.sync) if record.rtk is not None else None
    track = smooth_flight(record, rtk_off, cfg.smoothing)
    rep.add("Synchronisation", f"Telemetry is {offset.offset_s:+.2f} s from the video "
                               f"({offset.method.replace('_', ' ')}, {'reliable' if offset.reliable else 'uncertain'}); "
                               f"GPS smoothed ({track.source}).",
            {"offset_s": offset.offset_s, "correlation": offset.correlation,
             "median_gps_sigma_m": float(np.median(np.hypot(track.pos_std[:, 0], track.pos_std[:, 1])))},
            track.warnings)

    # 3 pre-flight gate + keyframes ----------------------------------------------------------
    metrics = frame_metrics(iter_frames(v, stride=sample_stride(v, cfg.gate)), cfg.gate)
    gate = evaluate_gate(metrics, v, offset, track, cfg.gate)
    rep.add("Pre-flight check", gate.summary().splitlines()[0],
            {c.name: c.status for c in gate.checks}, [c.message for c in gate.checks if c.status != "pass"])
    result = FrontResult(record, offset, track, gate, rep)
    if gate.decision == "reject" and not force:
        result.stopped = True
        return result

    ks = select_keyframes(iter_frames(v, stride=cfg.ingest.frame_stride), cfg.keyframes, track, offset.offset_s,
                          record.attitude)
    result.keyframes = ks
    rep.add("Keyframes", f"{len(ks.keyframes)} keyframes from {ks.frames_seen} frames.",
            {"keyframes": len(ks.keyframes), "forced": sum(k.forced for k in ks.keyframes)}, ks.warnings)
    if len(ks.keyframes) < 3:
        raise DataError("too few keyframes to reconstruct; the camera barely moved or the clip is too short")

    wanted = {k.index for k in ks.keyframes}
    images = {f.index: f.image for f in iter_frames(v) if f.index in wanted}
    result.images = [images[k.index] for k in ks.keyframes]
    t_video = np.array([k.t_video_s for k in ks.keyframes])
    pos, std = track.interpolate(t_video + offset.offset_s)
    result.gps_centers, result.gps_sigma = pos, np.maximum(std, 0.05)

    # 4 segmentation, features, pairs, matching ----------------------------------------------
    seg = make_segmenter(cfg.segmentation)
    result.segmentations = [seg.segment(im) for im in result.images]
    ext, mat = make_matching(cfg.matching)
    result.features = [ext.extract(im, s.exclusion_mask(cfg.segmentation.dilate_px))
                       for im, s in zip(result.images, result.segmentations)]
    n = len(result.images)
    desc = global_descriptors(result.features, cfg.pairing, cfg.seed) if n > cfg.pairing.sequence_window else None
    candidates = select_pairs(n, cfg.pairing, pos, desc)
    corr: dict[tuple[int, int], PairCorrespondences] = {}
    for p in candidates:
        f0, f1 = result.features[p.i], result.features[p.j]
        vp = verify_pair(f0, f1, mat.match(f0, f1), cfg.matching, cfg.seed)
        if vp.F is None:
            continue
        result.pairs.append(PairMatches(p.i, p.j, vp.matches.idx0, vp.matches.idx1))
        corr[(p.i, p.j)] = PairCorrespondences(f0.keypoints[vp.matches.idx0], f1.keypoints[vp.matches.idx1])
    warn = []
    if seg.is_mock:
        warn.append("segmentation is the placeholder: moving objects are not masked yet")
    if ext.is_fallback:
        warn.append("matching uses the SIFT fallback (LightGlue not installed)")
    rep.add("Matching", f"{len(result.pairs)} of {len(candidates)} candidate pairs verified.",
            {"median_features": float(np.median([len(f) for f in result.features])),
             "verified_pairs": len(result.pairs)}, warn)

    # 5 self-calibration ---------------------------------------------------------------------
    cal = self_calibrate(list(corr.values()), record.intrinsics, cfg.calibration, cfg.seed)
    K = cal.intrinsics
    rep.add("Calibration", f"Camera focal length {K.fx:.0f} px ({K.source}).",
            {"focal_px": K.fx, "focal_rel_std": cal.focal_rel_std}, cal.warnings)

    # 6 pose: AI track + preview, classical track, verifier ----------------------------------
    frames_idx = np.array([k.index for k in ks.keyframes])
    consecutive = {i: corr[(i, i + 1)] for i in range(n - 1) if (i, i + 1) in corr}
    ai = make_pose_model(cfg.pose).estimate(PoseInputs(frames_idx, t_video, K, consecutive, pos))
    preview = t0_preview(ai, pos, cfg.pose)
    write_preview(run, preview, title)
    start = _classical_start(record, preview, t_video + offset.offset_s, pos, cfg)
    recon = reconstruct(result.keypoints, result.pairs, start, pos, result.gps_sigma, cfg.classical)
    ver = verify(preview, recon, result.keypoints, pos, result.gps_sigma, cfg.verifier)
    write_accepted(run, ver, recon.sparse, title)
    result.reconstruction, result.verification = recon, ver
    rep.add("Camera poses", ver.reasons[0],
            {"accepted": ver.accepted, "reproj_p90_px": (ver.ai if ver.accepted == "ai" else ver.classical).reproj_p90_px,
             "gps_rms_m": recon.gps_rms_m, "sparse_points": len(recon.sparse.points),
             "classical_start": start.source},
            list(ai.warnings) + list(recon.warnings) + list(ver.reasons[1:]),
            outputs=[run / "preview.json", run / "accepted.json"], root=run)

    # 7 sparse model coverage, fed back to the gate ------------------------------------------
    cov = analyse_coverage(ver.trajectory, recon.sparse, result.keypoints, cfg.coverage)
    result.coverage, result.gate = cov, with_coverage(gate, cov)
    rep.add("Coverage", cov.checks[0].message, {"weak_fraction": cov.weak_fraction, "gaps": len(cov.regions)},
            [c.message for c in cov.checks if c.status != "pass"])
    return result
