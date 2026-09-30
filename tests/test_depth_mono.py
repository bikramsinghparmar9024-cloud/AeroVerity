import numpy as np
import pytest

from aerometric.config import DepthConfig
from aerometric.depth.mono import (
    DepthPrediction,
    FrameObservations,
    align_depth,
    frame_observations,
    make_depth_model,
    predict_aligned,
)
from aerometric.pose.types import CameraPose, Trajectory
from aerometric.types import CameraIntrinsics, DataError

W, H, F = 1024, 768, 800.0
K = CameraIntrinsics(W, H, F, F, W / 2, H / 2, source="provided")
R = np.array([[1.0, 0, 0], [0, -0.9396926, -0.3420201], [0, 0.3420201, -0.9396926]])  # 20 deg oblique
POSE = CameraPose.from_center(R, np.array([0.0, 0.0, 100.0]))


def true_depth(w=512, h=384, box=True):
    """Depth of a ground plane z=0 with a 20 m tall box, seen from POSE, at work resolution."""
    s = w / W
    u, v = np.meshgrid((np.arange(w) + 0.5) / s, (np.arange(h) + 0.5) / s)
    rays_c = np.column_stack([(u.ravel() - K.cx) / F, (v.ravel() - K.cy) / F, np.ones(u.size)])
    rays_w = rays_c @ POSE.R_cw
    C = POSE.center
    t = (0 - C[2]) / rays_w[:, 2]
    if box:
        t_top = (20 - C[2]) / rays_w[:, 2]
        P = C + rays_w * t_top[:, None]
        on_roof = (np.abs(P[:, 0]) < 15) & (np.abs(P[:, 1] - 30) < 15)
        t = np.where(on_roof, t_top, t)
    return (t * rays_c[:, 2]).reshape(h, w)  # optical-axis depth = t * z_cam of ray


def sparse_obs(depth_map, n=400, seed=0, outliers=0.0):
    rng = np.random.default_rng(seed)
    h, w = depth_map.shape
    px = np.column_stack([rng.integers(0, w, n), rng.integers(0, h, n)])
    d = depth_map[px[:, 1], px[:, 0]].copy()
    bad = rng.random(n) < outliers
    d[bad] *= rng.uniform(0.3, 3.0, bad.sum())
    return FrameObservations((px + 0.5) * (W / w) - 0.5 * (W / w) + 0.0, d, np.arange(n))


def test_alignment_recovers_metric_depth_from_affine_inverse_prediction():
    D = true_depth()
    obs = sparse_obs(D, outliers=0.1)
    disparity = 0.37 / D + 0.002  # what a relative inverse-depth network would output
    rng = np.random.default_rng(1)
    pred = DepthPrediction((disparity * (1 + rng.normal(0, 0.01, D.shape))).astype(np.float32),
                           "inverse_depth", "net", False)
    al = align_depth(pred, obs, W)
    assert al.good and not al.is_mock and al.points_used == 400
    assert np.nanmedian(np.abs(al.depth - D) / D) < 0.02


def test_alignment_of_depth_kind_with_scale_and_shift():
    D = true_depth()
    pred = DepthPrediction((0.01 * D - 0.3).astype(np.float32), "depth", "net", False)
    al = align_depth(pred, sparse_obs(D), W)
    assert al.a == pytest.approx(100, rel=1e-3) and al.b == pytest.approx(30, rel=1e-2)
    assert np.nanmax(np.abs(al.depth - D)) < 0.05


def test_too_few_points_or_bad_prediction_is_flagged():
    D = true_depth()
    few = align_depth(DepthPrediction(D.astype(np.float32), "depth", "net", False), sparse_obs(D, n=10), W)
    assert not few.good and np.isnan(few.depth).all()
    noise = np.random.default_rng(2).uniform(1, 2, D.shape).astype(np.float32)
    bad = align_depth(DepthPrediction(noise, "depth", "net", False), sparse_obs(D), W)
    assert not bad.good  # error looks small (depth varies little) but the map carries no structure
    flipped = align_depth(DepthPrediction((200 - D).astype(np.float32), "depth", "net", False), sparse_obs(D), W)
    assert flipped.a < 0 and not flipped.good


def test_stand_in_is_labelled_and_reasonable_on_ground():
    model = make_depth_model()
    assert model.is_mock and model.name == "mock-sparse-interp"
    with pytest.raises(DataError, match="not available"):
        make_depth_model(DepthConfig(backend="depth-anything"))
    D = true_depth(box=False)
    image = np.zeros((H, W, 3), np.uint8)
    al = predict_aligned(model, image, sparse_obs(D, n=300))
    assert al.is_mock and al.depth.shape == (384, 512)
    assert np.nanmedian(np.abs(al.depth - D) / D) < 0.01  # smooth ground is easy to interpolate


def test_stand_in_blurs_structure_between_points():
    D = true_depth(box=True)
    al = predict_aligned(make_depth_model(), np.zeros((H, W, 3), np.uint8), sparse_obs(D, n=120, seed=3))
    err = np.abs(al.depth - D) / D
    assert np.nanpercentile(err, 99) > 0.05  # box edges are smeared: exactly what cross-view checks catch


def test_frame_observations_from_sparse_model():
    traj = Trajectory(np.array([0]), np.array([0.0]), (POSE,), np.array([True]), K, "enu", True, "scipy-ba", False)
    pts = np.array([[0.0, 30.0, 0.0], [5.0, 40.0, 20.0], [0.0, 0.0, 200.0]])  # last one is behind the camera
    tracks = (np.array([[0, 0]]), np.array([[0, 1]]), np.array([[0, 2]]))
    kps = [np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])]
    obs = frame_observations(traj, pts, tracks, kps, 0)
    assert obs.point_ids.tolist() == [0, 1]
    Xc = pts[0] @ POSE.R_cw.T + POSE.t_cw
    assert obs.depth[0] == pytest.approx(Xc[2])
