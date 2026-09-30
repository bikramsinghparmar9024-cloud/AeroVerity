import numpy as np
from scipy.spatial.transform import Rotation

from aerometric.pose.classical import ReconstructionResult, SparseModel
from aerometric.pose.types import CameraPose, Trajectory
from aerometric.pose.verifier import verify
from aerometric.types import CameraIntrinsics

W, H, F, N = 1920, 1080, 1400.0, 8
K = CameraIntrinsics(W, H, F, F, W / 2, H / 2, source="provided")


def setup(noise=0.4):
    rng = np.random.default_rng(0)
    cams = []
    for i in range(N):
        a = np.radians(i * 8)
        C = np.array([60 * np.sin(a), 60 * (1 - np.cos(a)), 100.0])
        R_wc = Rotation.from_euler("zyx", [i * 8, 0, 160], degrees=True).as_matrix()
        cams.append(CameraPose.from_center(R_wc.T, C))
    X = np.column_stack([rng.uniform(-60, 90, 1500), rng.uniform(-40, 110, 1500), rng.uniform(0, 20, 1500)])
    kps = [[] for _ in range(N)]
    tracks = []
    for x in X:
        obs = []
        for c, p in enumerate(cams):
            Xc = p.R_cw @ x + p.t_cw
            uv = Xc[:2] / Xc[2] * F + [W / 2, H / 2]
            if Xc[2] > 1 and 0 < uv[0] < W and 0 < uv[1] < H:
                obs.append([c, len(kps[c])])
                kps[c].append(uv + rng.normal(0, noise, 2))
        if len(obs) >= 2:
            tracks.append(np.array(obs))
    kps = [np.array(k) for k in kps]
    gps = np.array([c.center for c in cams]) + rng.normal(0, 0.8, (N, 3))
    sigma = np.full((N, 3), 1.0)
    return cams, kps, tracks, gps, sigma


def traj(poses, valid=None, source="x", is_mock=False, world="enu"):
    return Trajectory(np.arange(N), np.arange(N) * 0.2, tuple(poses),
                      np.ones(N, bool) if valid is None else valid, K, world, world == "enu", source, is_mock)


def classical(cams, tracks):
    return ReconstructionResult(traj(cams, source="scipy-ba"),
                                SparseModel(np.zeros((len(tracks), 3)), tuple(tracks), np.zeros(len(tracks))),
                                0.5, 0.8)


def perturb(cams, rot_deg, pos_m, seed=5):
    rng = np.random.default_rng(seed)
    out = []
    for p in cams:
        dR = Rotation.from_rotvec(np.radians(rot_deg) * rng.normal(size=3) / np.sqrt(3)).as_matrix()
        out.append(CameraPose.from_center(dR @ p.R_cw, p.center + rng.normal(0, pos_m, 3)))
    return out


def test_good_ai_track_is_accepted():
    cams, kps, tracks, gps, sigma = setup()
    res = verify(traj(perturb(cams, 0.02, 0.05), source="net"), classical(cams, tracks), kps, gps, sigma)
    assert res.accepted == "ai" and res.trajectory.source == "net"
    assert res.ai.reproj_p90_px < 3 and res.ai.tracks_used > 500
    assert res.reasons[0].startswith("Using the AI track")


def test_inaccurate_ai_track_falls_back_with_reason():
    cams, kps, tracks, gps, sigma = setup()
    res = verify(traj(perturb(cams, 1.5, 0.5), source="mock-feedforward", is_mock=True),
                 classical(cams, tracks), kps, gps, sigma)
    assert res.accepted == "classical" and res.trajectory.source == "scipy-ba"
    assert "stand-in AI track failed" in res.reasons[0] and "reprojection" in res.reasons[0]
    assert res.classical.reproj_p90_px < res.ai.reproj_p90_px


def test_gps_disagreement_rejects_ai():
    cams, kps, tracks, gps, sigma = setup()
    shifted = [CameraPose.from_center(p.R_cw, p.center + [15, 0, 0]) for p in cams]  # rigid 15 m shift
    res = verify(traj(shifted), classical(cams, tracks), kps, gps, sigma)
    assert res.accepted == "classical"
    assert res.ai.reproj_p90_px < 3  # images are still consistent...
    assert "from GPS" in res.reasons[0]  # ...but it is in the wrong place


def test_points_behind_the_camera_fail_the_check_instead_of_becoming_nan():
    cams, kps, tracks, gps, sigma = setup()
    flipped = [CameraPose(np.diag([1.0, -1.0, -1.0]) @ np.diag([1.0, -1.0, -1.0]) @ p.R_cw, p.t_cw) if i % 2 == 0
               else CameraPose(np.diag([-1.0, 1.0, -1.0]) @ p.R_cw, p.t_cw) for i, p in enumerate(cams)]
    res = verify(traj(flipped), classical(cams, tracks), kps, gps, sigma)
    assert np.isfinite(res.ai.reproj_p90_px) and res.ai.reproj_p90_px > 3
    assert res.accepted == "classical"


def test_placeholders_or_local_frame_reject_ai():
    cams, kps, tracks, gps, sigma = setup()
    valid = np.ones(N, bool)
    valid[3] = False
    assert verify(traj(cams, valid), classical(cams, tracks), kps, gps, sigma).accepted == "classical"
    local = verify(traj(cams, world="local"), classical(cams, tracks), kps, gps, sigma)
    assert local.accepted == "classical" and "not placed in metric ENU" in local.reasons[0]
