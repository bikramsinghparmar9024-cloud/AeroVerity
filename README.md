AeroVerity 3D

> A single-pass drone video to georeferenced 3D reconstruction system with confidence-aware geometry.

**Team:** AIT_LogShield  
**Team ID:** 136059  
**Problem Statement:** SIH 2026 - 26158, *Single-Pass Drone Video to Accurate 3D Model Generation System*  
**Theme:** Robotics and Drones

## Overview

AeroVerity 3D turns drone video and flight telemetry into a metric, georeferenced 3D reconstruction. The pipeline is designed to work from a single flight pass and reports how well each part of the model is supported by visual evidence.

Instead of treating every reconstructed surface as equally reliable, the system labels geometry as:

- **MEASURED** - supported by consistent evidence across views; measurement is enabled.
- **INFERRED** - plausible geometry with incomplete or uncertain visual support.
- **UNKNOWN** - unsupported geometry that should not be used for measurement.

These labels help operators distinguish a visually complete model from one that is reliable for measurement.

## Goals

- Reconstruct useful 3D geometry from a single drone video pass.
- Align video frames with GPS, gyro, IMU, and barometer telemetry.
- Verify camera poses and geometry using both visual and flight data.
- Estimate uncertainty and expose it to operators.
- Produce geospatial, visualization, and analysis outputs.
- Reduce dependence on repeated flights and dense ground control point (GCP) surveys.

## Conceptual Pipeline

1. **Input preparation** - Ingest drone video and telemetry. Use optical flow and flight motion to estimate timing offsets, smooth the trajectory, assess video quality, and select keyframes.
2. **Feature matching and pose estimation** - Match keyframe features with ALIKED or DISK and LightGlue. Run a fast AI pose estimate alongside a GPS-constrained pycolmap/SfM track.
3. **Geometry and verification** - Compare depth across nearby views, check reprojection and GPS residuals, and fuse supported geometry using Open3D TSDF or Poisson reconstruction.
4. **Trust and outputs** - Classify surfaces as MEASURED, INFERRED, or UNKNOWN; generate model and geospatial outputs, semantic labels, and a quality report.

A re-fly advisor is proposed to identify coverage gaps and suggest viewpoints when another flight is possible.

## Architecture (Proposed)

```text
Drone video + flight telemetry
              |
      Input preparation
  (FFmpeg, telemetry parser,
   quality gate, keyframes)
              |
       Feature matching
    (ALIKED/DISK + LightGlue)
              |
      Pose estimation tracks
       /                 \
Fast AI pose        GPS-constrained SfM
       \                 /
        Pose and GPS verifier
              |
   Cross-view depth validation
              |
     Geometry fusion (Open3D)
              |
 Trust labels + uncertainty + QA
              |
 Mesh / point cloud / geospatial exports
```

This is a conceptual architecture from the project proposal. Implementation details and interfaces may change as the project develops.

## Planned Capabilities

- Synchronize video frames with GPS and motion telemetry.
- Reject or flag low-quality frames using blur, exposure, compression, synchronization, and feature-density checks.
- Use loop closure and viewpoint diversity to improve reconstruction.
- Verify AI-estimated trajectories against reprojection and GPS checks, with a classical SfM fallback.
- Compare depth estimates across views to reduce unsupported or distorted geometry.
- Estimate per-point or per-surface confidence with bootstrap runs and held-out-view error.
- Remove dynamic objects, sky, and water where possible, and assign semantic labels such as roads, buildings, and vegetation.
- Produce a quality report with relative and absolute accuracy information.

## Planned Outputs

Depending on the implemented modules, the system is intended to export:

- 3D meshes and point clouds: **GLB**, **PLY**, and **LAZ**
- 3D visualization: **3D Tiles** and **Gaussian Splatting**
- Geospatial products: **DSM**, **orthomosaic**, and **GeoTIFF**
- Semantic layers, change-detection results, and a signed quality report

Not every format or capability may be available in the current prototype.

## Technology Mentioned in the Proposal

| Area | Proposed tools |
| --- | --- |
| Video ingestion | FFmpeg |
| Feature extraction and matching | ALIKED or DISK, LightGlue |
| Structure from motion | COLMAP / pycolmap |
| Geometry processing and fusion | Open3D (TSDF / Poisson) |
| Semantic segmentation | SAM 3 (listed in the proposal) |
| Multi-view stereo | OpenMVS (listed in the proposal) |
| User interface | React, Tailwind CSS (listed in the proposal) |

The proposal does not specify pinned versions, deployment targets, or the final division of functionality among these components. Update this table to reflect the actual implementation and licenses before release.

## Repository Status

This README describes the proposed system. The source PDF does not include the implementation repository, confirmed setup commands, API endpoints, sample data, or test instructions. Replace the placeholders below as those become available.

```text
[ repository layout to be added ]
```

## Getting Started

Setup instructions are not included in the proposal. Add the verified steps for your repository here, for example:

```bash
# TODO: Clone the repository
# TODO: Install the documented dependencies
# TODO: Configure input video and telemetry
# TODO: Run the reconstruction pipeline
```

### Inputs

The proposed workflow expects:

- A drone video file.
- Associated flight telemetry, such as GPS and available gyro/IMU/barometer data.
- Optional DEM or imagery anchors to support georeferencing.

Specify supported file formats, coordinate reference systems, camera metadata requirements, and telemetry schemas when implemented.

## Accuracy and Limitations

A single flight may not capture hidden sides, underhangs, or other occluded areas. Such regions should remain INFERRED or UNKNOWN unless additional evidence supports them. Standard drone GPS and unsynchronized telemetry may also limit absolute positioning. The proposal addresses these issues through telemetry alignment, trajectory smoothing, robust similarity alignment, cross-view verification, and uncertainty reporting, but achieved accuracy must be measured on real data and reported with the relevant conditions.

The confidence labels indicate evidence support; they are not, by themselves, a guarantee of survey-grade accuracy. Report internal/relative accuracy separately from absolute/georeferencing accuracy.

## Roadmap

- [ ] Implement video and telemetry ingestion and synchronization.
- [ ] Add quality gating and keyframe selection.
- [ ] Integrate feature matching and pose estimation tracks.
- [ ] Add pose verification and fallback behavior.
- [ ] Implement cross-view depth checks and geometry fusion.
- [ ] Add confidence labeling and calibrated uncertainty reporting.
- [ ] Add semantic processing and export formats.
- [ ] Validate accuracy and document supported operating conditions.
