import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronerecon.geo import LocalFrame
from dronerecon.georef.gravity import GravityObservations, gravity_from_attitude, up_from_gimbal
from dronerecon.georef.similarity import georeference
from dronerecon.pose.types import CameraPose, Similarity, Trajectory
from dronerecon.types import AttitudeSeries, CameraIntrinsics

K = CameraIntrinsics(1920, 1080, 1400, 1400, 960, 540, source="provided")
FRAME = LocalFrame(28.6, 77.2, 200.0)
N = 12


def gimbal_rotation(yaw, pitch, roll):
    """Camera R_cw in ENU for heading `yaw` (deg from east, CCW), gimbal pitch and roll."""
    fwd = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0.0])
    left = np.array([-fwd[1], fwd[0], 0.0])
    R_level = np.array([-left, [0, 0, -1.0], fwd])  # rows: cam x (right), y (down), z (fwd)
    p = np.radians(pitch)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    r = np.radians(roll)
    Rz = np.array([[np.cos(r), np.sin(r), 0], [-np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Rx @ R_level


def truth(straight=True):
    cams, pitch, roll = [], [], []
    for i in range(N):
        C = np.array([i * 12.0, 0.4 * np.sin(i) if straight else 40 * np.sin(i / 3), 100.0])
        p, r = -60 + 3 * np.sin(i), 2 * np.cos(i)
        cams.append(CameraPose.from_center(gimbal_rotation(10.0, p, r), C))
        pitch.append(p)
        roll.append(r)
    return cams, np.array(pitch), np.array(roll)


def as_traj(poses):
    return Trajectory(np.arange(N) * 6, np.arange(N) * 0.5, tuple(poses), np.ones(N, bool), K,
                      "enu", True, "scipy-ba", False)


def distort(cams, roll_about_track_deg=3.0):
    """Input model = truth with a rigid tilt about the flight line (what GPS cannot see)."""
    R = Rotation.from_rotvec(np.radians(roll_about_track_deg) * np.array([1.0, 0, 0])).as_matrix()
    c = np.mean([p.center for p in cams], axis=0)
    sim = Similarity(1.0, R, c - R @ c)
    return as_traj([sim.apply_pose(p) for p in cams])


def rot_err(a, b):
    return np.degrees(np.linalg.norm(Rotation.from_matrix(a @ b.T).as_rotvec()))


def gps_of(cams, sigma=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([p.center for p in cams]) + rng.normal(0, sigma, (N, 3)), np.full((N, 3), sigma)


def test_up_from_gimbal_matches_constructed_cameras():
    for yaw, pitch, roll in [(0, -90, 0), (30, -45, 0), (120, -60, 5), (-70, -20, -8)]:
        R = gimbal_rotation(yaw, pitch, roll)
        expected = R @ [0, 0, 1.0]
        got = up_from_gimbal(np.array([pitch]), np.array([roll]))[0]
        assert got == pytest.approx(expected, abs=1e-9)


def test_rotation_from_gimbal_is_consistent_with_gravity():
    from dronerecon.georef.gravity import rotation_from_gimbal

    for yaw, pitch, roll in [(0, -90, 0), (90, -80, 0), (200, -45, 6), (315, -20, -8)]:
        R = rotation_from_gimbal(yaw, pitch, roll)
        assert R @ R.T == pytest.approx(np.eye(3), abs=1e-12) and np.linalg.det(R) == pytest.approx(1.0)
        assert R @ [0, 0, 1.0] == pytest.approx(up_from_gimbal(np.array([pitch]), np.array([roll]))[0], abs=1e-9)
    fwd = rotation_from_gimbal(90, 0, 0)[2]  # level camera facing east
    assert fwd == pytest.approx([1, 0, 0], abs=1e-12)


def test_straight_flight_tilt_is_unconstrained_without_gravity():
    cams, _, _ = truth()
    gps, sig = gps_of(cams)
    reg = georeference(distort(cams), np.zeros((0, 3)), gps, sig, FRAME)
    assert not reg.tilt_constrained
    assert any("tilt across the flight line" in w for w in reg.warnings)
    assert max(rot_err(a.R_cw, b.R_cw) for a, b in zip(reg.trajectory.poses, cams)) > 2.0


def test_gimbal_gravity_fixes_the_tilt():
    cams, pitch, roll = truth()
    gps, sig = gps_of(cams)
    rng = np.random.default_rng(1)
    att = AttitudeSeries(np.arange(N) * 0.5, np.column_stack([np.zeros(N), pitch + rng.normal(0, 0.3, N),
                                                              roll + rng.normal(0, 0.3, N)]))
    grav = gravity_from_attitude(att, np.arange(N) * 0.5, sigma_deg=0.5)
    reg = georeference(distort(cams), np.zeros((0, 3)), gps, sig, FRAME, grav)
    assert reg.tilt_constrained and reg.anchors == ("gps", "gravity")
    assert max(rot_err(a.R_cw, b.R_cw) for a, b in zip(reg.trajectory.poses, cams)) < 0.5
    assert reg.gravity_rms_deg < 1.0 and not reg.warnings


def test_wrong_roll_sign_is_reported():
    cams, pitch, roll = truth()
    roll = roll * 4  # make the sign matter
    cams = [CameraPose.from_center(gimbal_rotation(10.0, p, r), c.center) for c, p, r in zip(cams, pitch, roll)]
    gps, sig = gps_of(cams)
    grav = GravityObservations(np.arange(N), up_from_gimbal(pitch, roll, roll_sign=-1.0), np.full(N, 0.5))
    reg = georeference(as_traj(cams), np.zeros((0, 3)), gps, sig, FRAME, grav)
    assert any("roll sign" in w for w in reg.warnings)


def test_gps_outliers_are_left_out_and_points_move_with_poses():
    cams, _, _ = truth(straight=False)
    clean, sig = gps_of(cams)
    bad = clean.copy()
    bad[3] += [30, 0, 0]
    bad[8] += [0, -25, 5]
    # input model in an arbitrary local frame: scaled, rotated, shifted
    R = Rotation.from_euler("z", 40, degrees=True).as_matrix()
    to_local = Similarity(0.1, R, np.array([5.0, -3.0, 1.0]))
    X = np.array([[20.0, 5.0, 0.0], [60.0, -10.0, 3.0]])
    local = Trajectory(np.arange(N), np.arange(N) * 0.5, tuple(to_local.apply_pose(p) for p in cams),
                       np.ones(N, bool), K, "local", False, "scipy-ba", False)

    ref = georeference(local, to_local.apply_points(X), clean, sig, FRAME)
    reg = georeference(local, to_local.apply_points(X), bad, sig, FRAME)
    assert set(np.flatnonzero(reg.gps_outliers)) == {3, 8}
    assert reg.gps_rms_m < 1.0
    assert reg.points == pytest.approx(ref.points, abs=0.3)  # bad fixes barely move the answer
    assert reg.trajectory.world == "enu" and reg.trajectory.metric
    assert reg.tilt_constrained  # a curving path fixes orientation from GPS alone

    # Level flight at 100 m with 0.5 m GPS noise: tilt is known to a few tenths of a degree,
    # which is ~1 m on the ground 100 m below. The reported uncertainty must say so.
    tilt = max(rot_err(a.R_cw, b.R_cw) for a, b in zip(ref.trajectory.poses, cams))
    assert tilt < 3 * np.linalg.norm(ref.rotation_std_deg)
    assert reg.points == pytest.approx(X, abs=1.5)
