import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from aerometric.config import ClassicalConfig
from aerometric.pose.classical import reconstruct
from aerometric.pose.tracks import PairMatches, build_tracks
from aerometric.pose.types import CameraPose, Trajectory
from aerometric.types import CameraIntrinsics, DataError

W, H, F = 1920, 1080, 1400.0
K = CameraIntrinsics(W, H, F, F, W / 2, H / 2, source="provided")
N = 10


def truth(curved=False):
    rng = np.random.default_rng(0)
    cams = []
    for i in range(N):
        if curved:  # 90-degree turn of radius 70 m
            a = np.radians(i * 10)
            C = np.array([70 * np.sin(a), 70 * (1 - np.cos(a)), 100.0 + np.cos(i)])
            yaw = i * 10.0
        else:
            C = np.array([i * 12.0, 3.0 * np.sin(i / 2), 100.0 + np.cos(i)])
            yaw = 4 * np.sin(i)
        R_wc = Rotation.from_euler("zyx", [yaw, 2 * np.cos(i), 160], degrees=True).as_matrix()
        cams.append(CameraPose.from_center(R_wc.T, C))
    X = np.column_stack([rng.uniform(-40, 150, 3000), rng.uniform(-90, 90, 3000), rng.uniform(0, 25, 3000)])
    return cams, X


def observe(cams, X, noise=0.5, outlier_frac=0.0, seed=1):
    """Per-image keypoints (shuffled) and pairwise matches for pairs within 3 frames."""
    rng = np.random.default_rng(seed)
    kps, ids = [], []
    for p in cams:
        Xc = X @ p.R_cw.T + p.t_cw
        uv = Xc[:, :2] / Xc[:, 2:3] * F + [W / 2, H / 2]
        ok = (Xc[:, 2] > 1) & (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)
        idx = rng.permutation(np.flatnonzero(ok))
        kps.append(uv[idx] + rng.normal(0, noise, (len(idx), 2)))
        ids.append(idx)
    pairs = []
    for i in range(N):
        for j in range(i + 1, min(N, i + 4)):
            common, a, b = np.intersect1d(ids[i], ids[j], return_indices=True)
            if outlier_frac:
                bad = rng.random(len(b)) < outlier_frac
                b = b.copy()
                b[bad] = rng.integers(0, len(ids[j]), bad.sum())
            pairs.append(PairMatches(i, j, a, b))
    return kps, pairs


def initial_from(cams, rot_err_deg=1.0, pos_err_m=2.0, seed=3):
    rng = np.random.default_rng(seed)
    poses = []
    for p in cams:
        dR = Rotation.from_rotvec(np.radians(rot_err_deg) * rng.normal(size=3) / np.sqrt(3)).as_matrix()
        poses.append(CameraPose.from_center(dR @ p.R_cw, p.center + rng.normal(0, pos_err_m, 3)))
    return Trajectory(np.arange(N) * 6, np.arange(N) * 0.2, tuple(poses), np.ones(N, bool), K,
                      "enu", True, "mock-feedforward", True)


def rot_err(a, b):
    return np.degrees(np.linalg.norm(Rotation.from_matrix(a @ b.T).as_rotvec()))


def gps_for(cams, sigma=1.0, seed=4):
    rng = np.random.default_rng(seed)
    return np.array([c.center for c in cams]) + rng.normal(0, sigma, (N, 3)), np.full((N, 3), sigma)


def test_build_tracks_merges_and_rejects_conflicts():
    pairs = [PairMatches(0, 1, np.array([0, 1]), np.array([5, 6])),
             PairMatches(1, 2, np.array([5, 6]), np.array([9, 3])),
             PairMatches(0, 2, np.array([1]), np.array([4]))]  # conflicts with 1->6->3
    tracks = build_tracks([2, 7, 10], pairs)
    as_sets = [set(map(tuple, t.tolist())) for t in tracks]
    assert {(0, 0), (1, 5), (2, 9)} in as_sets
    assert not any((0, 1) in s for s in as_sets)  # the inconsistent track is dropped


def test_metric_track_improves_on_initial_poses():
    cams, X = truth()
    kps, pairs = observe(cams, X)
    gps, sigma = gps_for(cams)
    init = initial_from(cams)
    res = reconstruct(kps, pairs, init, gps, sigma)
    traj = res.trajectory
    assert traj.source == "scipy-ba" and not traj.is_mock and traj.metric and traj.world == "enu"
    c_err = np.linalg.norm(traj.centers - [c.center for c in cams], axis=1)
    init_c = np.linalg.norm(init.centers - [c.center for c in cams], axis=1)
    assert np.median(c_err) < 0.8 < np.median(init_c)
    assert max(relative_rot_errs(traj.poses, cams)) < 0.1
    assert res.rms_px < 1.0
    assert len(res.sparse.points) > 1000
    assert any("nearly straight" in w for w in res.warnings)  # absolute roll left to IMU gravity


def relative_rot_errs(est, cams):
    return [rot_err(est[k].R_cw @ est[0].R_cw.T, cams[k].R_cw @ cams[0].R_cw.T) for k in range(1, N)]


def test_curved_flight_fixes_absolute_orientation():
    cams, X = truth(curved=True)
    X[:, 0] -= 30  # keep the scene under the turn
    kps, pairs = observe(cams, X)
    gps, sigma = gps_for(cams)
    res = reconstruct(kps, pairs, initial_from(cams), gps, sigma)
    assert not any("nearly straight" in w for w in res.warnings)
    # absolute orientation now comes from GPS alone (1 m noise over a 70 m turn: a few tenths of a degree)
    assert max(rot_err(p.R_cw, c.R_cw) for p, c in zip(res.trajectory.poses, cams)) < 0.75


def test_outlier_matches_are_removed():
    cams, X = truth()
    kps, pairs = observe(cams, X, outlier_frac=0.05)
    gps, sigma = gps_for(cams)
    res = reconstruct(kps, pairs, initial_from(cams), gps, sigma)
    assert res.rms_px < 1.2
    assert np.median(np.linalg.norm(res.trajectory.centers - [c.center for c in cams], axis=1)) < 1.0


def test_invalid_initial_pose_is_recovered():
    cams, X = truth()
    kps, pairs = observe(cams, X)
    gps, sigma = gps_for(cams)
    init = initial_from(cams)
    valid = np.ones(N, bool)
    valid[5] = False
    broken = Trajectory(init.frame_indices, init.t_video_s, init.poses, valid, K, "enu", True, "x", True)
    res = reconstruct(kps, pairs, broken, gps, sigma)
    assert max(relative_rot_errs(res.trajectory.poses, cams)) < 0.1


def test_backend_and_input_checks():
    cams, X = truth()
    kps, pairs = observe(cams, X)
    gps, sigma = gps_for(cams)
    init = initial_from(cams)
    with pytest.raises(DataError, match="not installed"):
        reconstruct(kps, pairs, init, gps, sigma, ClassicalConfig(backend="pycolmap"))
    local = Trajectory(init.frame_indices, init.t_video_s, init.poses, init.valid, K, "local", False, "x", True)
    with pytest.raises(DataError, match="ENU"):
        reconstruct(kps, pairs, local, gps, sigma)
