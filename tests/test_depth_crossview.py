import numpy as np
import pytest

from dronerecon.config import DepthConfig
from dronerecon.depth.crossview import score_all, score_frame
from dronerecon.depth.mono import AlignedDepth, FrameObservations, make_depth_model, predict_aligned
from dronerecon.pose.types import CameraPose, Trajectory
from dronerecon.types import CameraIntrinsics, DataError

W, H, F = 1024, 768, 800.0
K = CameraIntrinsics(W, H, F, F, W / 2, H / 2, source="provided")
SW, SH = 256, 192  # work resolution
S = W / SW
NADIR = np.diag([1.0, -1.0, -1.0])


def poses(spacing=12.0, n=5):
    return [CameraPose.from_center(NADIR, np.array([i * spacing, 0.0, 100.0])) for i in range(n)]


def traj(ps):
    return Trajectory(np.arange(len(ps)), np.arange(len(ps)) * 1.0, tuple(ps), np.ones(len(ps), bool), K,
                      "enu", True, "scipy-ba", False)


def ray_depth(pose, box=False):
    """True optical-axis depth at work resolution: ground z=0, optional 20 m box roof."""
    v, u = np.mgrid[0:SH, 0:SW].astype(float)
    rays_c = np.column_stack([(u.ravel() * S - K.cx) / F, (v.ravel() * S - K.cy) / F, np.ones(u.size)])
    rays_w = rays_c @ pose.R_cw
    C = pose.center
    t = -C[2] / rays_w[:, 2]
    if box:
        tt = (20 - C[2]) / rays_w[:, 2]
        P = C + rays_w * tt[:, None]
        roof = (np.abs(P[:, 0] - 24) < 12) & (np.abs(P[:, 1]) < 12)
        t = np.where(roof, tt, t)
    return (t * rays_c[:, 2]).reshape(SH, SW).astype(np.float32)


def aligned(depth, good=True):
    return AlignedDepth(depth, S, 1.0, 0.0, "depth", 0.0, 100, good, "test", False)


def test_true_depths_agree():
    ps = poses()
    depths = [aligned(ray_depth(p)) for p in ps]
    sc = score_frame(traj(ps), depths, 2)
    assert sc.neighbours == (0, 1, 3, 4)
    assert sc.checked_fraction > 0.9
    assert sc.agreeing_fraction() > 0.99


def test_wrong_patch_is_found():
    ps = poses()
    depths = [aligned(ray_depth(p)) for p in ps]
    bad = depths[2].depth.copy()
    bad[60:120, 90:160] *= 1.06  # 6 % too deep: e.g. a smeared roof
    depths[2] = aligned(bad)
    sc = score_frame(traj(ps), depths, 2)
    patch = sc.agreement[65:115, 95:155]
    rest = np.concatenate([sc.agreement[:50].ravel(), sc.agreement[130:].ravel()])
    assert np.nanmean(patch) < 0.05
    assert np.nanmean(rest) > 0.98


def test_near_identical_viewpoints_give_no_evidence():
    ps = poses(spacing=0.5)  # 0.5 m apart at 100 m: < 2 degrees
    depths = [aligned(ray_depth(p)) for p in ps]
    sc = score_frame(traj(ps), depths, 2)
    assert np.isnan(sc.agreement).mean() > 0.95


def test_stand_in_depth_disagrees_at_structure_edges():
    ps = poses()
    rng = np.random.default_rng(0)
    model = make_depth_model()
    depths = []
    for p in ps:
        D = ray_depth(p, box=True)
        px = np.column_stack([rng.integers(0, SW, 250), rng.integers(0, SH, 250)])
        obs = FrameObservations(px * S, D[px[:, 1], px[:, 0]].astype(float), np.arange(250))
        depths.append(predict_aligned(model, np.zeros((H, W, 3), np.uint8), obs, DepthConfig(work_width=SW)))
    assert all(d.good for d in depths)
    sc = score_all(traj(ps), depths)[2]
    truth = ray_depth(ps[2], box=True)
    edge = np.zeros_like(truth, bool)
    gy, gx = np.gradient(truth)
    edge[np.hypot(gx, gy) > 1.0] = True
    import cv2
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    flat = ~cv2.dilate(edge.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    assert np.nanmean(sc.agreement[edge]) < np.nanmean(sc.agreement[flat]) - 0.3


def test_unusable_reference_is_refused():
    ps = poses()
    depths = [aligned(ray_depth(p)) for p in ps]
    depths[2] = aligned(depths[2].depth, good=False)
    with pytest.raises(DataError, match="no usable depth"):
        score_frame(traj(ps), depths, 2)
    assert score_all(traj(ps), depths)[2] is None
