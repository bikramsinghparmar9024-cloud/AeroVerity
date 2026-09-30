# dronerecon

Single-pass drone video → georeferenced, metric 3D model that says how much of itself it can vouch for.
Built for Smart India Hackathon 2026, problem statement **SIH26158** (NTRO).

Every surface in the model carries one of three labels:

| Label | Meaning | Can you measure on it? |
|---|---|---|
| **Measured** | Seen well enough from enough angles; calibrated uncertainty ≤ 0.25 m | Yes, with a stated ± |
| **Inferred** | Surface is there, but uncertainty is larger (≤ 1 m) or could not be estimated | No |
| **Unknown** | Not seen clearly, or a moving object | No |

Where the model is weak, the **re-fly advisor** says why and where to point the camera next time.

## Install

Python 3.10+, runs fully offline.

```bash
pip install -e .[dev]
pip install "git+https://github.com/cvg/LightGlue.git"   # recommended: learned feature matching
```

LightGlue downloads its weights on first use (`aliked-n16.pth`, `aliked_lightglue_v0-1_arxiv.pth`,
about 50 MB) into the torch cache. On an air-gapped machine, copy those two files into
`~/.cache/torch/hub/checkpoints/` beforehand. Without LightGlue, SIFT matching is used automatically.

## Easiest: the local web app

```bash
python -m dronerecon app --runs runs
```

Open **http://localhost:8765**, press **New reconstruction**, choose the video and its `.SRT` (or flight-log
`.CSV`), and press **Upload and start**. The page shows each of the 15 steps as it completes, fills in the 3D
view as soon as camera positions exist, and lists the quality report and downloads when the run finishes.
Earlier runs stay in the run picker. Runs are processed one at a time; everything stays on this computer.

## Reconstruct a flight from the command line

```bash
python -m dronerecon run --video DJI_0001.MP4 --telemetry DJI_0001.SRT --out runs/flight1 --viewer
```

`--viewer` opens a live page on `http://127.0.0.1:8765`: a quick preview first, then the verified result.

Optional inputs, each improving accuracy:

| Option | What it adds |
|---|---|
| `--intrinsics camera.json` | Camera calibration (otherwise estimated, or assumed 84° field of view) |
| `--imu imu.csv`, `--baro baro.csv` | Gyro for time sync, barometer for height and tilt |
| `--rtk flight.pos` | RTK/PPK positions (RTKLIB `.pos`) instead of standard GPS |
| `--dem dem.asc` / `.tif` | Elevation model (WGS84) to check height and horizontal position |
| `--satellite ortho.tif` / `.png` + `.pgw` | Georeferenced imagery to check horizontal position |
| `--config settings.json` | Override any setting in `configs/default.json` |
| `--key signing.pem` | Signing key for the report (created if missing; keep it private) |

The run stops early if the pre-flight check expects an unusable model (heavy blur, wrong telemetry…);
`--force` continues anyway.

## What you get (`runs/flight1/`)

| File | Contents |
|---|---|
| `report/quality_report.html` | Readable report: trust summary, every step, warnings, reliability diagram |
| `report/quality_report.json` | Same, hash-chained and Ed25519-signed |
| `products/model.glb`, `model.ply` | Mesh; each vertex carries trust, uncertainty (m) and class |
| `products/points.las` | Coloured point cloud, WGS84, ASPRS classes, trust in User Data |
| `products/tiles/tileset.json` | 3D Tiles 1.1 for Cesium and similar web viewers |
| `products/dsm.tif`, `ortho.tif` | Surface model and orthophoto (GeoTIFF, WGS84) |
| `products/trust.tif`, `class.tif`, `sigma.tif` | Trust, class and uncertainty layers |
| `products/*.geojson` | Class and trust areas as polygons |
| `products/splat_seed.ply` | Gaussian-splat file **seeded, not trained** (see limitations) |
| `products/refly_advice.json` | Weak areas with suggested camera positions |

## Compare two flights

```bash
python -m dronerecon compare runs/flight1 runs/flight2 --out runs/change
```

Aligns flight 2 onto flight 1, then reports only height changes larger than both flights' uncertainty:
`change_dz.tif`, `change_class.tif`, `change_areas.geojson` (areas and volumes), and a signed report.

## Check a report has not been altered

```bash
python -m dronerecon verify runs/flight1/report/quality_report.json --files runs/flight1
```

## Current limitations

Honest status: the full pipeline runs end to end and is tested on **synthetic** flights; it has not yet
been run on real footage. Some components are stand-ins until their models or packages are installed —
the report and viewer always say when a stand-in was used.

| Component | Now | Intended |
|---|---|---|
| Fast AI pose track | Stand-in (two-view chaining) | Feed-forward reconstruction network |
| Monocular depth | Stand-in (interpolated sparse depth) | Depth network |
| Segmentation | **DeepLabV3** for people, vehicles, animals (if its weights are cached) + colour rules for vegetation/water; no roads or buildings yet; small objects from high altitude are often missed | Aerial semantic segmentation network |
| Feature matching | **ALIKED + LightGlue** (installed; SIFT is the automatic fallback) | — |
| Bundle adjustment, fusion | Built in (SciPy, scikit-image) | pycolmap, Open3D (faster, Poisson option) |
| LAZ, trained splats | `.las` only; splat seed only | laspy[lazrs]; gsplat on a GPU |

## Tests

```bash
python -m pytest -q
```

About 210 tests; the end-to-end ones render a synthetic flight and take a few minutes.
