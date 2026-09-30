# dronerecon — project handoff (read this instead of scanning the code)

SIH 2026 problem statement **SIH26158** (NTRO): *Single-pass drone video → georeferenced, metrically
accurate, textured 3D model*. Python package `dronerecon` in `src/`, ~110 files, **225 tests pass
(2 skipped)** as of 2026-09-30. Everything runs offline on CPU (Windows 11, Python 3.10.8).

## How to work with the user (important)
- The architecture below is **final**. Don't redesign, add or remove stages, or change the data flow.
- Build **one module at a time**, test it, then stop. Report in this format:
  **IMPLEMENTED / FILES / VALIDATION / NEXT**. Be concise and token-efficient; don't re-explain the architecture.
- **Ask before any download or install** (pip packages, model weights), stating the source and size.
  Only LightGlue has been approved and installed so far.
- The user wants the UI to be decent, professional and human-feeling.
- Don't commit. The folder sits inside a git repo rooted at the user's home dir with no commits.
- Be honest: say when something is a stand-in, untested or synthetic-only.
- The user has **no real drone footage yet**. The "Measured" threshold stays at **0.25 m** (user's choice).

## Run / test (PowerShell)
```powershell
$env:PYTHONPATH="src"                              # package is NOT pip-installed yet
python -m dronerecon app --runs runs               # web app on http://localhost:8765 (upload + queue + viewer)
python -m dronerecon run --video V --telemetry V.SRT [--intrinsics cam.json] --out runs/x [--viewer] [--config f.json]
python -m dronerecon compare runs/a runs/b --out runs/change
python -m dronerecon verify runs/x/report/quality_report.json --files runs/x
python -m pytest -q                                # full suite ~10 min (end-to-end tests render synthetic flights)
```
- The Browser-pane preview is defined in `.claude/launch.json` as `dronerecon-app` (the web app), `dronerecon-demo`
  and `dronerecon-viewer`. Start it with `preview_start name=dronerecon-app`.
- Restarting the app **interrupts a running job** (it gets marked failed/"Interrupted"), so ask before restarting.
- The page (`index.html`) is read on every request, so HTML edits need no restart. Python edits do.

## Architecture (final) → where each stage lives
```
INPUTS  video, flight SRT/CSV, IMU, baro, RTK/PPK .pos, intrinsics .json
 1 ingest        ingest/video.py (ffprobe→OpenCV fallback), telemetry.py (DJI SRT old/new, CSV, IMU, baro),
                 rtk.py (RTKLIB .pos), intrinsics.py (json: fx/fy/cx/cy | camera_matrix+dist_coeffs | hfov_deg),
                 flight.py → FlightRecord
 2 sync          sync/offset.py  flow↔GPS-speed (0.5 s window) / gyro cross-correlation; t_tel = t_video + offset;
                                 ambiguity check; RTK offset
                 sync/smoothing.py  6-state CV Kalman + RTS, chi² gating, stale-fix drop → SmoothedTrack (ENU)
 3 gate+keyframes quality/gate.py (blur, exposure, blockiness, texture, sync, coverage, GPS) → proceed/warn/reject
                 quality/keyframes.py (LK parallax 8% width, overlap, baseline, yaw; sharpest candidate)
 4 seg+matching  segmentation/base.py (Label enum, exclusion mask, make_segmenter: auto|deeplab|mock)
                 segmentation/deeplab.py (torchvision DeepLabV3 VOC: person/vehicle/animal + colour rules)
                 matching/features.py (auto: ALIKED+LightGlue, else SIFT fallback; MAGSAC verify)
                 matching/pairs.py (sequence + GPS radius + VLAD retrieval, GPS gate)
 5 self-calib    calibration.py (Sampson-error LSQ for f,k1,k2 + observability check; rolling-shutter assessment)
 6 pose          pose/types.py (CameraPose R_cw/t_cw OpenCV conv., Trajectory, Similarity)
                 pose/feedforward.py  AI track = **STAND-IN** (two-view chaining) + t0_preview (Umeyama to GPS)
                 pose/classical.py    GPS-constrained scipy bundle adjustment ("scipy-ba"); robust loss on image
                                      residuals only; mean-rotation gauge prior; tracks.py (union-find, DLT)
                 pose/verifier.py     AI accepted only if p90 reproj ≤3 px, GPS nRMS ≤3σ, no placeholders
                 viewer/ (server.py, export.py, static/index.html, jobs.py)  live viewer + web app
 7 sparse+coverage quality/coverage.py (ground grid: points/views/angle, weak regions → gate feedback)
 8 georef        georef/similarity.py (robust similarity: GPS + gravity + baro + known distances; tilt check)
                 georef/gravity.py (gimbal → up vector, rotation_from_gimbal), dem.py (.asc/.tif WGS84),
                 satellite.py (ortho keyframes vs GeoTIFF/.pgw imagery)
 9 depth         depth/mono.py  depth model = **STAND-IN** (interpolated sparse depth); robust affine alignment
                                (depth or inverse depth), correlation/scale checks
                 depth/crossview.py  forward-backward consistency → per-pixel agreement score (evidence only)
10 fusion        fusion/semantic.py (occlusion-checked label voting, dynamic flag)
                 fusion/tsdf.py (built-in TSDF + skimage marching cubes; drops unobserved-voxel phantoms;
                                 Poisson → "needs Open3D" error)
11 uncertainty   uncertainty/bootstrap.py (keyframe subsets + GPS perturbation; pose_run re-triangulates
                                           original tracks; surface_run re-aligns depth + re-fuses)
                 uncertainty/calibration.py (held-out-view errors, chi-dist scale factor, reliability SVG)
12 trust         trust.py  effective σ = calibrated σ × inflation(agreement); ≤0.25 m MEASURED, ≤1 m INFERRED,
                           else/unobserved/dynamic UNKNOWN; measure_distance only between MEASURED
13 re-fly        refly.py (cluster weak vertices + coverage gaps; reason; two viewpoints; heading/pitch/lat-lon)
14 outputs       outputs/mesh_io.py (GLB + PLY with _TRUST/_SIGMA/_CLASS), points_io.py (LAS 1.2 WGS84,
                 splat SEED not trained), rasters.py (DSM/ortho/trust/class/sigma GeoTIFF EPSG:4326),
                 tiles.py (3D Tiles 1.1 + GeoJSON layers), report.py (hash-chained Ed25519-signed report)
15 change        change/register.py (GPS prior + ortho feature prior + trimmed point-to-plane ICP),
                 change/changemap.py (dz vs combined σ, not-comparable, areas/volumes, GeoTIFF+GeoJSON)
WIRING           pipeline/front.py (phases 1–7), back.py (8–14, saves products/state.npz), compare.py (15)
                 cli.py (run/compare/verify/viewer/app, ConsoleReport, cp1252-safe output)
SHARED           types.py (VideoInfo, GpsTrack, CameraIntrinsics, FlightRecord…), config.py (all *Config
                 dataclasses; configs/default.json must equal the defaults, which a test checks), geo.py
                 (LocalFrame equirectangular ENU; exact WGS84 lla_to_ecef, enu_to_ecef_matrix)
```

## Key conventions
- Camera: OpenCV (x right, y down, z forward); `X_c = R_cw X_w + t_cw`. World = local ENU metres; the origin is the
  first smoothed GPS fix (`SmoothedTrack.frame`, a `LocalFrame`). All products convert through `LocalFrame.to_lla`.
- Gimbal: yaw is a compass heading (0 = N, clockwise); pitch 0 = level, −90 = nadir; roll sign is configurable
  (`georef.gimbal_roll_sign`, unverified on real DJI data).
- The classical BA starts from **GPS + gimbal attitude** when attitude exists, otherwise from the T0 preview.
- Every stage adds a `ReportBuilder` entry; the report is finalised and signed in `run_back`.
- JSON sent to the browser must be strict: use `outputs.report.json_safe` (NaN → null).

## Web app (viewer/)
- `jobs.py`: a JobQueue runs one job at a time on a worker thread. Each run lives in `runs/<id>/`, with
  `input/video.*`, `telemetry.*`, `intrinsics.json` and `status.json` (read with utf-8-sig).
- `server.py` routes: `/api/config`, `/api/state?run=`, `/api/runs` (GET/POST),
  `PUT /api/runs/<id>/files/<video|telemetry|intrinsics>?name=`, `POST /api/runs/<id>/start`,
  `/runs/<id>/report`, `/runs/<id>/download/<name>` (fixed whitelist in `jobs.DOWNLOADS`). Localhost only.
- `index.html` is a single self-contained file (canvas 3D view, run picker, New reconstruction dialog, 15-step
  progress, downloads). It uses no CDN (air-gapped).

## Status: done vs not done
- All 15 phases are built, wired and tested **on synthetic data only**. **Never run on real footage** (biggest risk).
- Stand-ins: the AI pose model and monocular depth. Segmentation has no road/building classes. No LAZ
  (laspy not installed), no trained splats (gsplat/GPU), no pycolmap/Open3D (built-in scipy BA + TSDF used).
- Speed: a 40 s 1280×720 flight takes roughly 30–60 min on CPU. Uncertainty was sped up on 2026-09-30:
  `n_runs` 5, `bootstrap_voxel_factor` 2.0, `bootstrap_max_points` 3000. Only the uncertainty estimate is
  slightly noisier; the model itself is unchanged. The accuracy impact vs ground truth has **not** been measured yet.

## Demo data
- `demo_data/residential/`: synthetic 40 s residential flight rendered by `tools/render_residential.py`
  (height-field ray-march, 3.5 s/frame). Files: `flight.mp4` (1280×720, 15 fps, mp4v), `flight.srt` (DJI
  format, per frame, GPS with ~0.45 m drift), `camera_calibration.json` (OpenCV model with distortion), and
  `ground_truth.csv` (scoring only; do NOT upload it as telemetry).
- `demo_data/flight.avi|.srt|camera.json|fast.json`: small flat-ground synthetic flight. `fast.json` sets
  voxel 0.5 and n_runs 3.
- `runs/demo`, `runs/<timestamp>-*`: earlier runs visible in the app.
- Test helper `tests/synthetic_flight.py` renders a flight with known truth for the end-to-end tests.

## Environment quirks (save time)
- The sandbox blocks `raw.githubusercontent.com`; github.com release assets and git clone work. LightGlue weights
  (`aliked-n16.pth`, `aliked_lightglue_v0-1_arxiv.pth`) are already in `~/.cache/torch/hub/checkpoints/`, along with
  `deeplabv3_resnet50_coco-cd0a2569.pth`.
- There's no ffmpeg (OpenCV is used), no GDAL/rasterio/pyproj (tifffile writes GeoTIFFs), and `cryptography` 50 is installed.
- PowerShell 5.1: `Set-Content -Encoding utf8` writes a BOM; `&&` isn't available. Redirected console output is
  cp1252 (cli handles it).
- The auto-mode command classifier sometimes fails transiently; retry, or do read-only work meanwhile.
- OpenCV `remap` is limited to 32767 lookups per map dimension; use scipy `map_coordinates` instead.

## Next steps (in priority order)
1. Run a **real DJI clip + SRT** (user to supply) and fix what breaks.
2. Measure the fast-uncertainty impact on `demo_data/residential` against `ground_truth.csv`.
3. With the user's approval: a depth network (Depth Anything V2 Small, ~100 MB), then Open3D/pycolmap, then laspy[lazrs].
4. Add a "continue anyway" (force) option to the upload dialog; the gate can reject compressed video.
5. Demo deck/script for judging; `pip install -e .` (may fetch setuptools, so ask first).
