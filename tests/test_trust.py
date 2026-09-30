import numpy as np
import pytest

from dronerecon.depth.crossview import CrossViewScore
from dronerecon.depth.mono import AlignedDepth
from dronerecon.pose.types import CameraPose, Trajectory
from dronerecon.trust import TrustLabel, assess, measure_distance, vertex_agreement
from dronerecon.types import CameraIntrinsics

M, I, U = TrustLabel.MEASURED, TrustLabel.INFERRED, TrustLabel.UNKNOWN


def test_single_threshold_rule():
    sigma = np.array([0.1, 0.2, 0.6, 3.0, np.nan, 0.1, 0.1, 0.1])
    agree = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, np.nan, 1.0])
    weight = np.array([5, 5, 5, 5, 5, 5, 5, 0.1])
    t = assess(sigma, agree, weight)
    assert t.labels.tolist() == [M, M, I, U, I, I, M, U]
    # disagreement widens 0.1 m to 0.3 m (-> Inferred); unscored widens it to 0.15 m (still Measured)
    assert t.effective_sigma_m[5] == pytest.approx(0.3)
    assert t.effective_sigma_m[6] == pytest.approx(0.15)
    assert t.reasons[3] == "uncertainty too large"
    assert "too few runs" in t.reasons[4]
    assert t.reasons[7] == "not observed clearly enough"
    assert t.reasons[0] == ""


def test_moving_objects_are_unknown_and_mock_is_reported():
    t = assess(np.array([0.05, 0.05]), np.array([1.0, 1.0]), np.array([9.0, 9.0]),
               dynamic=np.array([False, True]), evidence_is_mock=True)
    assert t.labels.tolist() == [M, U] and t.reasons[1] == "moving object"
    assert t.evidence_is_mock and "stand-in" in t.warnings[0]


def test_measurement_only_between_measured_points():
    verts = np.array([[0.0, 0, 0], [3.0, 4.0, 0], [10.0, 0, 0]])
    t = assess(np.array([0.1, 0.2, 0.8]), np.ones(3), np.ones(3) * 5)
    ok = measure_distance(t, verts, 0, 1)
    assert ok.allowed and ok.value_m == pytest.approx(5.0) and ok.sigma_m == pytest.approx(np.hypot(0.1, 0.2))
    no = measure_distance(t, verts, 0, 2)
    assert not no.allowed and np.isnan(no.value_m)
    assert "point 2 is Inferred" in no.reason and "both points to be Measured" in no.reason


def test_area_weighted_fractions():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [0, 10, 0]], float)
    faces = np.array([[0, 1, 2], [0, 3, 4]])  # small face all Measured, big face touches an Unknown vertex
    t = assess(np.array([0.1, 0.1, 0.1, 0.1, 5.0]), np.ones(5), np.ones(5) * 5)
    f = t.fractions(verts, faces)
    assert f["measured"] == pytest.approx(0.5 / 50.5) and f["unknown"] == pytest.approx(50 / 50.5)
    assert sum(t.fractions().values()) == pytest.approx(1.0)


def test_vertex_agreement_samples_visible_views():
    K = CameraIntrinsics(400, 300, 300, 300, 200, 150, source="provided")
    pose = CameraPose.from_center(np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 50.0]))
    traj = Trajectory(np.array([0]), np.array([0.0]), (pose,), np.array([True]), K, "enu", True, "x", False)
    depth = AlignedDepth(np.full((150, 200), 50.0, np.float32), 2.0, 1, 0, "depth", 0, 99, True, "t", False)
    ag = np.full((150, 200), np.nan)
    ag[:, :100] = 1.0  # left half agreed, right half disagreed
    ag[:, 100:] = 0.0
    score = CrossViewScore(ag, np.ones((150, 200), int), np.ones((150, 200), int), (1,))
    verts = np.array([[-20.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 0.0, 20.0], [500.0, 0, 0]])
    a = vertex_agreement(verts, traj, [depth], [score])
    assert a[0] == 1.0 and a[1] == 0.0
    assert np.isnan(a[2])  # 30 m in front of the surface the camera saw: not visible there
    assert np.isnan(a[3])  # outside the frame
