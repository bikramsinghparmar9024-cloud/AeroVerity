"""Pipeline configuration. Values live in JSON files; defaults are defined here once."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised for unknown keys or invalid configuration values."""


@dataclass(frozen=True)
class PathsConfig:
    work_dir: str = "runs"
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"


@dataclass(frozen=True)
class IngestConfig:
    default_hfov_deg: float = 84.0  # used only when no intrinsics are supplied
    frame_stride: int = 1
    rtk_source: str = "ppk"  # label for .pos tracks: "rtk" or "ppk"
    rtk_accept_quality: tuple[int, ...] = (1, 2)  # RTKLIB Q: 1 fix, 2 float, 5 single


@dataclass(frozen=True)
class TelemetryConfig:
    """CSV column overrides. None means auto-detect from common header names."""

    time_col: str | None = None
    lat_col: str | None = None
    lon_col: str | None = None
    alt_col: str | None = None
    gyro_unit: str = "rad/s"  # "rad/s" or "deg/s"
    drop_no_fix: bool = True  # drop samples with lat == lon == 0


@dataclass(frozen=True)
class SyncConfig:
    max_lag_s: float = 5.0  # search window for the video↔telemetry offset
    grid_dt_s: float = 0.02  # resampling step for cross-correlation
    min_correlation: float = 0.5  # below this the offset is reported as unreliable
    min_overlap: float = 0.5  # share of the shorter signal that must overlap at a lag
    ambiguity_margin: float = 0.05  # a rival peak this close to the best makes the result unreliable
    peak_exclusion_s: float = 0.5  # rival peaks must be at least this far from the best lag
    flow_width: int = 320  # frames are downscaled to this width for optical flow
    max_corners: int = 300


@dataclass(frozen=True)
class SmoothingConfig:
    accel_noise_h: float = 1.5  # m/s^2, white-noise acceleration (horizontal)
    accel_noise_v: float = 0.8  # m/s^2 (vertical)
    gnss_sigma_h_m: float = 1.5  # per-axis 1-sigma when the source reports no accuracy
    gnss_sigma_v_m: float = 3.0
    gate_chi2: float = 16.27  # 3-DOF 99.9 % innovation gate; larger jumps are outliers
    drop_stale: bool = True  # drop repeated identical fixes (logged faster than GPS updates)


@dataclass(frozen=True)
class GateConfig:
    analysis_width: int = 640  # frames are downscaled to this width for blur/exposure/features
    sample_fps: float = 2.0  # how many frames per second of video the gate inspects
    min_sharpness: float = 60.0  # variance of Laplacian at analysis_width
    dark_level: int = 8
    bright_level: int = 247
    max_clipped_fraction: float = 0.25  # share of pixels crushed to black or blown to white
    min_brightness: float = 30.0  # mean grey level range considered usable
    max_brightness: float = 225.0
    max_blockiness: float = 1.35  # 8x8 block-edge / interior gradient ratio
    min_features: int = 150  # corners per frame at analysis_width
    warn_bad_fraction: float = 0.2  # share of frames failing a check before we warn
    fail_bad_fraction: float = 0.5  # ... and before we reject the flight
    warn_sync_coverage: float = 0.9  # share of video time covered by telemetry
    fail_sync_coverage: float = 0.5
    warn_gps_sigma_m: float = 5.0  # median smoothed horizontal 1-sigma


@dataclass(frozen=True)
class KeyframeConfig:
    work_width: int = 640
    min_sharpness: float = 60.0  # variance of Laplacian at work_width
    target_parallax: float = 0.08  # median feature shift since last keyframe, as share of width
    candidate_parallax: float = 0.05  # frames past this are eligible; the sharpest one wins
    min_survival: float = 0.5  # share of keyframe features still tracked
    min_tracks: int = 30
    max_baseline_m: float = 0.0  # 0 disables; otherwise force a keyframe every N metres
    max_yaw_deg: float = 15.0  # camera yaw change that forces a keyframe
    max_corners: int = 500


@dataclass(frozen=True)
class SegmentationConfig:
    backend: str = "auto"  # "auto" (DeepLabV3 if cached, else placeholder), "deeplab" or "mock"
    work_width: int = 1024  # images are segmented at this width, labels resized back
    dilate_px: int = 5  # grow excluded regions so edge features on moving objects are dropped


@dataclass(frozen=True)
class MatchingConfig:
    backend: str = "auto"  # "auto" (LightGlue if installed, else OpenCV), "lightglue", "opencv"
    extractor: str = "aliked"  # LightGlue backend only: "aliked" or "disk"
    max_keypoints: int = 4096
    max_dim: int = 1600  # longest image side used for extraction
    ratio: float = 0.8  # OpenCV fallback: Lowe ratio test
    ransac_px: float = 1.5  # epipolar inlier threshold at full resolution
    min_inliers: int = 30  # fewer verified matches than this and the pair is dropped


@dataclass(frozen=True)
class PairingConfig:
    sequence_window: int = 5  # pair each keyframe with the next N keyframes
    gps_radius_m: float = 30.0  # camera centres closer than this are paired
    gps_max_neighbors: int = 10
    retrieval_top_k: int = 5  # loop-closure candidates per image
    retrieval_min_similarity: float = 0.25
    retrieval_gps_gate_m: float = 150.0  # reject retrieval pairs farther apart than this; 0 disables
    vlad_clusters: int = 32


@dataclass(frozen=True)
class CalibrationConfig:
    refine_provided: bool = False  # re-estimate even when intrinsics were supplied
    max_pairs: int = 30
    max_matches_per_pair: int = 300
    prior_focal_rel_sigma: float = 0.3  # for a FOV-derived prior; provided intrinsics use 0.02
    prior_dist_sigma: float = 0.2  # prior on k1, k2 (zero mean)
    robust_scale_px: float = 1.0
    max_focal_rel_std: float = 0.05  # above this the focal length is treated as unobservable
    readout_s: float = 0.0  # rolling-shutter readout time from the camera spec; 0 = unknown
    typical_readout_s: float = 0.03  # used only to warn when the readout is unknown
    rs_threshold_px: float = 1.0  # image motion during readout above this makes RS matter


@dataclass(frozen=True)
class PoseConfig:
    ai_backend: str = "mock"  # feed-forward pose model; only the labelled stand-in exists so far
    min_pair_matches: int = 30  # consecutive keyframes with fewer matches break the chain
    preview_min_gps_points: int = 3  # GPS fixes needed to place the T0 preview in ENU
    straight_track_ratio: float = 0.1  # track sideways spread / length below this leaves roll loose


@dataclass(frozen=True)
class ClassicalConfig:
    backend: str = "scipy"  # "scipy" (built-in bundle adjustment) or "pycolmap" (not installed)
    min_tri_angle_deg: float = 1.5  # tracks seen from nearly the same direction are skipped
    max_reproj_px: float = 4.0  # observations above this after adjustment are dropped
    robust_scale_px: float = 1.0
    rotation_prior_deg: float = 5.0  # weak pull toward initial orientation (keeps roll defined)
    straight_track_ratio: float = 0.1  # below this sideways/length ratio, warn that roll is loose
    max_points: int = 20000  # longest tracks are kept first
    max_nfev: int = 100


@dataclass(frozen=True)
class VerifierConfig:
    max_reproj_p90_px: float = 3.0  # 90th-percentile reprojection error the AI track must meet
    max_gps_nrms: float = 3.0  # RMS of (camera centre - GPS) / GPS sigma
    min_valid_fraction: float = 1.0  # share of AI poses that must be real (not placeholders)
    max_tracks: int = 3000  # tracks re-triangulated per trajectory


@dataclass(frozen=True)
class CoverageConfig:
    cell_m: float = 0.0  # ground grid cell; 0 = auto (1/40 of the median footprint width)
    min_cell_m: float = 0.5
    min_points_per_cell: int = 3
    min_views_per_cell: int = 3  # distinct keyframes observing the cell's points
    min_angle_deg: float = 3.0  # median triangulation angle of the cell's points
    min_region_cells: int = 4  # smaller gaps are not reported as regions
    warn_low_fraction: float = 0.2  # share of imaged ground with weak coverage
    fail_low_fraction: float = 0.6
    image_grid: tuple[int, int] = (8, 6)  # columns, rows for per-keyframe point spread
    min_image_spread: float = 0.3  # share of image cells holding at least one point


@dataclass(frozen=True)
class GeorefConfig:
    gps_robust_sigmas: float = 2.0  # soft-L1 scale for GPS residuals, in units of their sigma
    gps_outlier_sigmas: float = 4.0  # residual beyond this marks a GPS fix as an outlier
    gravity_sigma_deg: float = 1.0  # default uncertainty of a gravity observation
    baro_sigma_m: float = 0.3  # barometric altitude noise (relative; its zero offset is estimated)
    rotation_prior_deg: float = 10.0  # weak pull toward the input orientation (keeps roll defined)
    max_roll_std_deg: float = 1.0  # above this the tilt across the flight line is called unconstrained
    gimbal_roll_sign: float = 1.0  # +1: positive gimbal roll = right side down (check on a real flight)
    dem_sigma_m: float = 3.0  # DEM height accuracy (SRTM-class); also the robust-loss scale
    dem_min_relief_m: float = 5.0  # below this terrain relief a horizontal shift is not estimated
    dem_min_points: int = 50
    sat_sigma_m: float = 3.0  # absolute accuracy of the satellite image's georeferencing
    sat_search_m: float = 30.0  # how far the model may be off horizontally
    sat_max_frames: int = 8  # keyframes compared with the satellite image
    sat_min_inliers: int = 25  # matches needed for a keyframe's shift to count
    sat_max_spread_m: float = 3.0  # keyframe shifts must agree within this (robust spread)


@dataclass(frozen=True)
class DepthConfig:
    backend: str = "auto"  # "auto" (Depth Anything V2 if its weights are on disk, else stand-in), "depth-anything", "mock"
    model: str = "depth-anything/Depth-Anything-V2-Small-hf"  # local model folder or Hugging Face id (cache only)
    work_width: int = 512  # depth maps are produced at this width
    min_points: int = 30  # sparse points needed in a frame to align its depth map
    huber_rel: float = 0.05  # robust-fit scale, as a share of the median sparse depth
    max_rel_residual: float = 0.15  # median |error| / depth above this marks the alignment poor
    min_correlation: float = 0.5  # prediction must track sparse depth (when depth varies at all)
    min_depth_variation: float = 0.02  # std/mean of sparse depth below which correlation is not checked
    neighbours: int = 4  # keyframes on each side checked against
    depth_tol: float = 0.01  # relative depth disagreement allowed between views
    reproj_tol_px: float = 1.0  # forward-backward reprojection error allowed (work pixels)
    min_angle_deg: float = 2.0  # views closer than this in angle add no evidence


@dataclass(frozen=True)
class SemanticConfig:
    occlusion_tol: float = 0.05  # a point is visible if its depth is within 5 % of the depth map
    min_votes: int = 1  # fewer votes leave the point unlabelled
    dynamic_share: float = 0.5  # vote share for person/vehicle/animal that marks a point as dynamic


@dataclass(frozen=True)
class FusionConfig:
    method: str = "tsdf"  # "tsdf" (built in) or "poisson" (needs Open3D, not installed)
    voxel_m: float = 0.0  # 0 = auto (about 2x the ground sampling distance of the depth maps)
    trunc_voxels: float = 3.0  # truncation distance in voxels
    max_voxels: int = 8_000_000  # voxel size grows if the scene would need more
    unscored_weight: float = 0.3  # weight of depth pixels no neighbour could check
    min_agreement: float = 0.5  # pixels whose neighbours mostly disagree are not fused
    chunk_voxels: int = 1_000_000


@dataclass(frozen=True)
class UncertaintyConfig:
    n_runs: int = 5
    bootstrap_voxel_factor: float = 2.0  # re-run fusion at this multiple of the main voxel size (speed)
    bootstrap_max_points: int = 3000  # longest tracks used by each re-run's bundle adjustment (speed)
    keep_fraction: float = 0.8  # share of keyframes kept in each run
    gps_perturb_scale: float = 1.0  # GPS noise added per run, in units of each fix's sigma
    min_valid_runs: int = 3  # points measured in fewer runs get no uncertainty
    max_gap_m: float = 2.0  # surface further than this from a vertex counts as missing in that run
    min_calibration_samples: int = 50  # fewer held-out errors than this and calibration is not trusted
    max_scale_warning: float = 2.0  # warn when bootstrap sigma is off by more than this factor
    reliability_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    reliability_bins: int = 8


@dataclass(frozen=True)
class TrustConfig:
    measured_sigma_m: float = 0.25  # effective 1-sigma at or below this: MEASURED
    inferred_sigma_m: float = 1.0  # at or below this: INFERRED; above: UNKNOWN
    disagreement_inflation: float = 2.0  # sigma x (1 + this * (1 - agreement))
    unscored_inflation: float = 1.5  # sigma multiplier where no neighbour could check the surface
    min_weight: float = 0.5  # fused observation weight below this: UNKNOWN
    occlusion_tol: float = 0.05  # visibility test when sampling agreement into 3-D


@dataclass(frozen=True)
class ReflyConfig:
    cluster_m: float = 8.0  # weak vertices closer than this form one region
    min_area_m2: float = 20.0  # smaller regions are not worth a re-fly
    max_regions: int = 5
    min_views: int = 3  # fewer existing views than this: "seen by too few frames"
    min_angle_deg: float = 5.0  # existing views spread less than this: "seen from one direction only"
    grazing_deg: float = 70.0  # views more oblique than this to the surface: "grazing angle"
    new_view_angle_deg: float = 12.0  # angle between the two proposed views (parallax)
    standoff_m: float = 0.0  # distance to target; 0 = median distance of the existing cameras
    min_altitude_agl_m: float = 30.0  # proposed cameras stay at least this far above the local ground


@dataclass(frozen=True)
class RasterConfig:
    cell_m: float = 0.0  # 0 = auto (twice the median mesh edge length)
    fill_holes_px: int = 2  # gaps up to this many cells are filled from neighbours
    nodata: float = -9999.0
    occlusion_tol: float = 0.05  # orthophoto: depth-map visibility tolerance


@dataclass(frozen=True)
class ChangeConfig:
    icp_max_iter: int = 50
    icp_trim: float = 0.7  # use the best-fitting share of correspondences (changed areas fall outside)
    icp_max_dist_m: float = 5.0  # correspondences further apart than this are ignored
    icp_tol_m: float = 1e-4  # stop when the update is smaller than this
    min_overlap: float = 0.3  # share of B points that must find a partner in A
    ortho_min_inliers: int = 30
    significance: float = 2.0  # |dz| must exceed this many combined sigmas (about 95 %)
    min_change_m: float = 0.5  # and at least this many metres
    min_area_m2: float = 10.0  # smaller change patches are dropped as noise
    fallback_sigma_m: float = 1.0  # used for Inferred cells whose uncertainty was not estimated


@dataclass(frozen=True)
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    keyframes: KeyframeConfig = field(default_factory=KeyframeConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    georef: GeorefConfig = field(default_factory=GeorefConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)
    refly: ReflyConfig = field(default_factory=ReflyConfig)
    raster: RasterConfig = field(default_factory=RasterConfig)
    change: ChangeConfig = field(default_factory=ChangeConfig)
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type[T], data: dict[str, Any], where: str) -> T:
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else None
        if is_dataclass(default):
            if not isinstance(value, dict):
                raise ConfigError(f"{where}.{name}: expected an object")
            kwargs[name] = _build(type(default), value, f"{where}.{name}")
        else:  # JSON arrays become tuples so configs stay immutable and comparable
            kwargs[name] = tuple(value) if isinstance(value, list) else value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load config from JSON, falling back to defaults for missing keys."""
    if path is None:
        return PipelineConfig()
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {p}") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{p}: invalid JSON ({exc})") from None
    cfg = _build(PipelineConfig, data, "config")
    if cfg.ingest.frame_stride < 1:
        raise ConfigError("config.ingest.frame_stride must be >= 1")
    if cfg.ingest.rtk_source not in ("rtk", "ppk"):
        raise ConfigError("config.ingest.rtk_source must be 'rtk' or 'ppk'")
    if cfg.telemetry.gyro_unit not in ("rad/s", "deg/s"):
        raise ConfigError("config.telemetry.gyro_unit must be 'rad/s' or 'deg/s'")
    return cfg
