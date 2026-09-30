import numpy as np
import pytest

from dronerecon.config import FusionConfig
from dronerecon.depth.crossview import score_all
from dronerecon.depth.mono import AlignedDepth
from dronerecon.fusion.tsdf import fuse_tsdf
from dronerecon.pose.types import CameraPose, Trajectory
from dronerecon.segmentation import Label, SegmentationResult
from dronerecon.types import CameraIntrinsics, DataError

W, H, F = 640, 480, 400.0
K = CameraIntrinsics(W, H, F, F, W / 2, H / 2, source="provided")
S = 0.5  # depth maps at half resolution
ROOF = (0.0, 0.0, 10.0, 12.0)  # centre e, centre n, half-size, height
CAR = (-20.0, 10.0, 2.5, 1.8)


def poses(n=6):
    out = []
    for i in range(n):
        tilt = np.radians(15)
        R = np.array([[1, 0, 0], [0, -np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), -np.cos(tilt)]])
        out.append(CameraPose.from_center(R, np.array([-35.0 + 14 * i, -25.0, 70.0])))
    return out


def traj(ps):
    return Trajectory(np.arange(len(ps)), np.arange(len(ps)) * 1.0, tuple(ps), np.ones(len(ps), bool), K,
                      "enu", True, "scipy-ba", False)


def _box_hit(C, rays, box):
    t = (box[3] - C[2]) / rays[:, 2]
    P = C + rays * t[:, None]
    return t, (np.abs(P[:, 0] - box[0]) < box[2]) & (np.abs(P[:, 1] - box[1]) < box[2])


def cast(pose, s=S):
    w, h = int(W * s), int(H * s)
    v, u = np.mgrid[0:h, 0:w].astype(float) / s
    rc = np.column_stack([(u.ravel() - K.cx) / F, (v.ravel() - K.cy) / F, np.ones(u.size)])
    rays = rc @ pose.R_cw
    C = pose.center
    t = -C[2] / rays[:, 2]
    lab = np.full(t.shape, Label.OTHER)
    for box, cls in ((ROOF, Label.BUILDING), (CAR, Label.VEHICLE)):
        tb, hit = _box_hit(C, rays, box)
        closer = hit & (tb < t)
        t = np.where(closer, tb, t)
        lab = np.where(closer, cls, lab)
    return (t * rc[:, 2]).reshape(h, w).astype(np.float32), lab.reshape(h, w).astype(np.uint8)


def scene():
    ps = poses()
    depths, segs = [], []
    for p in ps:
        d, _ = cast(p)
        depths.append(AlignedDepth(d, 1 / S, 1.0, 0.0, "depth", 0.0, 100, True, "test", False))
        _, lab = cast(p, s=1.0)
        segs.append(SegmentationResult(lab, None, "test", False))
    return traj(ps), depths, segs


def surface_error(v):
    """Distance to the nearest true horizontal surface (ground, roof top) for vertices above them."""
    on_roof = (np.abs(v[:, 0] - ROOF[0]) < ROOF[2] - 1) & (np.abs(v[:, 1] - ROOF[1]) < ROOF[2] - 1)
    return np.where(on_roof, np.abs(v[:, 2] - ROOF[3]), np.abs(v[:, 2]))


def test_fused_mesh_matches_ground_and_roof():
    t, depths, segs = scene()
    mesh = fuse_tsdf(t, depths, score_all(t, depths), segs, FusionConfig(voxel_m=0.5))
    assert mesh.backend == "builtin-tsdf" and len(mesh.faces) > 1000
    v = mesh.vertices
    flat = (np.abs(v[:, 0]) > ROOF[2] + 2) | (np.abs(v[:, 1]) > ROOF[2] + 2)
    roof = (np.abs(v[:, 0]) < ROOF[2] - 1) & (np.abs(v[:, 1]) < ROOF[2] - 1)
    assert roof.sum() > 50 and np.median(np.abs(v[roof, 2] - ROOF[3])) < 0.25
    ground = flat & (np.abs(v[:, 2]) < 3)
    assert np.median(np.abs(v[ground, 2])) < 0.25
    assert mesh.semantics is not None
    roof_labels = mesh.semantics.labels[roof]
    assert np.mean(roof_labels == Label.BUILDING) > 0.9


def test_moving_objects_are_left_out():
    t, depths, segs = scene()
    mesh = fuse_tsdf(t, depths, score_all(t, depths), segs, FusionConfig(voxel_m=0.5))
    v = mesh.vertices
    over_car = (np.abs(v[:, 0] - CAR[0]) < CAR[2] - 0.5) & (np.abs(v[:, 1] - CAR[1]) < CAR[2] - 0.5)
    assert not np.any(over_car & (v[:, 2] > 1.0))  # no car roof in the surface


def test_disagreeing_depth_is_not_fused():
    t, depths, segs = scene()
    bad = depths[2].depth.copy()
    bad[40:120, 60:200] += 6.0  # a wrong patch in one frame only
    depths[2] = AlignedDepth(bad, 1 / S, 1.0, 0.0, "depth", 0.0, 100, True, "test", False)
    mesh = fuse_tsdf(t, depths, score_all(t, depths), None, FusionConfig(voxel_m=0.5))
    v = mesh.vertices
    flat = (np.abs(v[:, 0]) > ROOF[2] + 2) | (np.abs(v[:, 1]) > ROOF[2] + 2)
    far_from_car = np.hypot(v[:, 0] - CAR[0], v[:, 1] - CAR[1]) > CAR[2] + 3
    assert np.percentile(np.abs(v[flat & far_from_car, 2]), 99) < 0.6


def test_voxel_size_is_capped_and_poisson_is_explicit():
    t, depths, _ = scene()
    mesh = fuse_tsdf(t, depths, None, None, FusionConfig(voxel_m=0.05, max_voxels=300_000))
    assert mesh.voxel_m > 0.05 and any("voxel size raised" in w for w in mesh.warnings)
    assert any("no cross-view scores" in w for w in mesh.warnings)
    with pytest.raises(DataError, match="Open3D"):
        fuse_tsdf(t, depths, cfg=FusionConfig(method="poisson"))


def test_needs_metric_enu():
    t, depths, _ = scene()
    local = Trajectory(t.frame_indices, t.t_video_s, t.poses, t.valid, K, "local", False, "x", False)
    with pytest.raises(DataError, match="metric ENU"):
        fuse_tsdf(local, depths)
