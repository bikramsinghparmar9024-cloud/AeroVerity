import numpy as np
import pytest

from dronerecon.config import CoverageConfig
from dronerecon.pose.classical import SparseModel
from dronerecon.pose.types import CameraPose, Trajectory
from dronerecon.quality.coverage import analyse_coverage, with_coverage
from dronerecon.quality.gate import CheckResult, FrameMetrics, GateReport
from dronerecon.types import CameraIntrinsics, DataError

W, H, F = 1600, 1200, 1200.0
K = CameraIntrinsics(W, H, F, F, W / 2, H / 2, source="provided")
NADIR = np.diag([1.0, -1.0, -1.0])  # camera z looks down, x east, y south


def trajectory(n=12, spacing=15.0, alt=100.0):
    poses = tuple(CameraPose.from_center(NADIR, np.array([i * spacing, 0.0, alt])) for i in range(n))
    return Trajectory(np.arange(n) * 6, np.arange(n) * 0.5, poses, np.ones(n, bool), K, "enu", True, "scipy-ba", False)


def observe(traj, X):
    """Tracks and keypoints for points X seen by every camera whose image contains them."""
    kps = [[] for _ in range(len(traj))]
    tracks, kept = [], []
    for x in X:
        obs = []
        for c, p in enumerate(traj.poses):
            Xc = p.R_cw @ x + p.t_cw
            u, v = Xc[:2] / Xc[2] * F + [W / 2, H / 2]
            if Xc[2] > 1 and 0 <= u < W and 0 <= v < H:
                obs.append([c, len(kps[c])])
                kps[c].append([u, v])
        if len(obs) >= 2:
            tracks.append(np.array(obs))
            kept.append(x)
    return SparseModel(np.array(kept), tuple(tracks), np.zeros(len(kept))), [np.array(k) for k in kps]


def ground_points(n=30000, hole=None, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-80, 250, n), rng.uniform(-80, 80, n), rng.uniform(-0.5, 0.5, n)])
    if hole is not None:
        (e0, e1), (n0, n1) = hole
        X = X[~((X[:, 0] > e0) & (X[:, 0] < e1) & (X[:, 1] > n0) & (X[:, 1] < n1))]
    return X


def test_dense_ground_is_well_covered():
    traj = trajectory()
    sparse, kps = observe(traj, ground_points())
    rep = analyse_coverage(traj, sparse, kps, CoverageConfig(cell_m=4.0, min_region_cells=6))
    assert rep.weak_fraction < 0.2
    assert {c.name: c.status for c in rep.checks} == {"Ground coverage": "pass", "Point spread": "pass"}
    # A single pass always under-covers the ground before the first and after the last frame.
    assert len(rep.regions) == 2
    assert all(r.center_en[0] < 0 or r.center_en[0] > 165 for r in rep.regions)
    # footprint of 12 nadir frames: (11*15 + 133) m x 100 m at 100 m altitude
    assert rep.imaged_area_m2 == pytest.approx(298 * 100, rel=0.1)


def test_gap_is_located():
    traj = trajectory()
    sparse, kps = observe(traj, ground_points(hole=((60, 100), (-20, 20))))  # 40 x 40 m hole
    rep = analyse_coverage(traj, sparse, kps, CoverageConfig(cell_m=4.0))
    inner = [r for r in rep.regions if 0 < r.center_en[0] < 165]
    assert len(inner) == 1
    assert inner[0].center_en == pytest.approx((80, 0), abs=4)
    assert inner[0].area_m2 == pytest.approx(1600, rel=0.35)
    assert "gaps in total" in rep.checks[0].message


def test_sparse_flight_warns_or_fails():
    traj = trajectory(n=4, spacing=60.0)  # little overlap: most ground seen by < 3 views
    sparse, kps = observe(traj, ground_points(8000))
    rep = analyse_coverage(traj, sparse, kps, CoverageConfig(cell_m=4.0))
    assert rep.weak_fraction > 0.6 and rep.checks[0].status == "fail"
    assert "Inferred or Unknown" in rep.checks[0].message


def test_point_spread_flags_one_sided_frames():
    traj = trajectory()
    X = ground_points()
    X = X[X[:, 1] > 35]  # only a northern strip (~15% of each frame) has texture
    sparse, kps = observe(traj, X)
    rep = analyse_coverage(traj, sparse, kps, CoverageConfig(cell_m=4.0))
    spread = {c.name: c for c in rep.checks}["Point spread"]
    assert spread.status == "warn" and "keyframe(s)" in spread.message


def test_auto_cell_size():
    traj = trajectory()
    sparse, kps = observe(traj, ground_points(5000))
    rep = analyse_coverage(traj, sparse, kps)
    assert rep.grid.cell_m == pytest.approx(133.3 / 40, rel=0.05)


def test_feedback_merges_into_gate():
    traj = trajectory(n=4, spacing=60.0)
    sparse, kps = observe(traj, ground_points(8000))
    rep = analyse_coverage(traj, sparse, kps, CoverageConfig(cell_m=4.0))
    m = FrameMetrics(*(np.zeros(1) for _ in range(7)))
    gate = GateReport("proceed", (CheckResult("Blur", "pass", "ok"),), m)
    merged = with_coverage(gate, rep)
    assert merged.decision == "reject"
    assert [c.name for c in merged.checks][:2] == ["Blur", "Ground coverage"]
    again = with_coverage(merged, rep)  # re-applying replaces, not duplicates
    assert len(again.checks) == len(merged.checks)


def test_requires_metric_enu():
    traj = trajectory()
    sparse, kps = observe(traj, ground_points(2000))
    local = Trajectory(traj.frame_indices, traj.t_video_s, traj.poses, traj.valid, K, "local", False, "x", False)
    with pytest.raises(DataError, match="metric ENU"):
        analyse_coverage(local, sparse, kps)
