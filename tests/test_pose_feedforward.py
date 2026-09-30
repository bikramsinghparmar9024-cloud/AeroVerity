import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronerecon.calibration import PairCorrespondences
from dronerecon.config import PoseConfig
from dronerecon.pose.feedforward import PoseInputs, make_pose_model, preview_payload, t0_preview
from dronerecon.pose.geometry import umeyama
from dronerecon.pose.types import CameraPose
from dronerecon.types import CameraIntrinsics, DataError

W, H, F = 1920, 1080, 1400.0
K = CameraIntrinsics(W, H, F, F, W / 2, H / 2, source="provided")


def true_cameras(n=10):
    """ENU world, z up. Oblique cameras flying east at 100 m with gentle yaw changes."""
    cams = []
    for i in range(n):
        C = np.array([i * 10.0, 2.0 * np.sin(i / 2), 100.0])
        R_wc = Rotation.from_euler("zyx", [3 * np.sin(i), 0, 180 - 20], degrees=True).as_matrix()
        cams.append(CameraPose.from_center(R_wc.T, C))
    return cams


def scene(seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-60, 160, 5000), rng.uniform(-120, 120, 5000),
                            rng.uniform(0, 20, 5000)])


def project(p, X):
    Xc = X @ p.R_cw.T + p.t_cw
    uv = Xc[:, :2] / Xc[:, 2:3] * F + [W / 2, H / 2]
    ok = (Xc[:, 2] > 1) & (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)
    return uv, ok


def consecutive(cams, X, noise=0.3, drop=()):
    rng = np.random.default_rng(1)
    out = {}
    for k in range(len(cams) - 1):
        if k in drop:
            continue
        a, oa = project(cams[k], X)
        b, ob = project(cams[k + 1], X)
        ok = oa & ob
        out[k] = PairCorrespondences(a[ok] + rng.normal(0, noise, (ok.sum(), 2)),
                                     b[ok] + rng.normal(0, noise, (ok.sum(), 2)))
    return out


def inputs(cams, gps, drop=()):
    n = len(cams)
    return PoseInputs(np.arange(n) * 6, np.arange(n) * 0.2, K, consecutive(cams, scene(), drop=drop), gps)


def rot_err_deg(Ra, Rb):
    return np.degrees(np.linalg.norm(Rotation.from_matrix(Ra @ Rb.T).as_rotvec()))


def test_umeyama_exact():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    R = Rotation.from_euler("xyz", [10, -30, 70], degrees=True).as_matrix()
    dst = 2.5 * src @ R.T + [1, 2, 3]
    sim = umeyama(src, dst)
    assert sim.s == pytest.approx(2.5) and np.allclose(sim.R, R) and np.allclose(sim.t, [1, 2, 3])


def test_stand_in_is_labelled():
    model = make_pose_model()
    assert model.is_mock and model.name == "mock-feedforward"
    with pytest.raises(DataError, match="not available"):
        make_pose_model(PoseConfig(ai_backend="vggt"))


def test_t0_preview_matches_truth_after_gps_placement():
    cams = true_cameras()
    rng = np.random.default_rng(2)
    gps = np.array([c.center for c in cams]) + rng.normal(0, 0.5, (len(cams), 3))
    traj = make_pose_model().estimate(inputs(cams, gps))
    assert traj.world == "local" and traj.is_mock and traj.valid.all()
    prev = t0_preview(traj, gps)
    assert prev.world == "enu" and prev.metric
    err = np.linalg.norm(prev.centers - [c.center for c in cams], axis=1)
    assert np.median(err) < 1.5
    # Relative rotations between keyframes are what the chain actually determines.
    rel = [rot_err_deg(prev.poses[k + 1].R_cw @ prev.poses[k].R_cw.T,
                       cams[k + 1].R_cw @ cams[k].R_cw.T) for k in range(len(cams) - 1)]
    assert max(rel) < 0.5
    # Absolute roll about a near-straight path is not constrained by GPS, and we say so.
    assert any("nearly straight" in w for w in prev.warnings)


def test_curved_track_fixes_absolute_orientation():
    cams = []
    for i in range(10):  # quarter-circle turn: GPS spans two horizontal directions
        a = np.radians(i * 10)
        C = np.array([80 * np.sin(a), 80 * (1 - np.cos(a)), 100.0])
        R_wc = Rotation.from_euler("zyx", [i * 10, 0, 160], degrees=True).as_matrix()
        cams.append(CameraPose.from_center(R_wc.T, C))
    gps = np.array([c.center for c in cams])
    prev = t0_preview(make_pose_model().estimate(inputs(cams, gps)), gps)
    assert not any("nearly straight" in w for w in prev.warnings)
    # ~1 deg of accumulated drift over 9 chained links is expected from the stand-in.
    assert max(rot_err_deg(p.R_cw, c.R_cw) for p, c in zip(prev.poses, cams)) < 2.0


def test_broken_link_is_flagged_not_hidden():
    cams = true_cameras()
    gps = np.array([c.center for c in cams])
    traj = make_pose_model().estimate(inputs(cams, gps, drop=(4,)))
    assert traj.valid.tolist() == [True] * 5 + [False] + [True] * 4
    assert any("placeholders" in w for w in traj.warnings)


def test_preview_payload_is_json_and_honest():
    cams = true_cameras(4)
    gps = np.array([c.center for c in cams])
    payload = preview_payload(t0_preview(make_pose_model().estimate(inputs(cams, gps)), gps))
    text = json.dumps(payload)
    back = json.loads(text)
    assert back["is_mock"] is True and back["source"] == "mock-feedforward"
    assert len(back["cameras"]) == 4 and back["world"] == "enu"
    assert any("stand-in" in n for n in back["notes"])


def test_preview_without_enough_gps_stays_local():
    cams = true_cameras(4)
    traj = make_pose_model().estimate(inputs(cams, None))
    gps = np.full((4, 3), np.nan)
    prev = t0_preview(traj, gps)
    assert prev.world == "local" and not prev.metric
    assert any("too few GPS" in w for w in prev.warnings)
