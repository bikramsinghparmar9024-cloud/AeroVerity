import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from aerometric.config import GeorefConfig
from aerometric.georef.similarity import KnownDistance, georeference
from aerometric.pose.types import CameraPose, Similarity, Trajectory
from test_georef_similarity import FRAME, K, gimbal_rotation, rot_err

N = 12


def level_flight(n=N, spacing=12.0, curve=40.0):
    cams = []
    for i in range(n):
        C = np.array([i * spacing, curve * np.sin(i / 3), 100.0 + 2 * np.sin(i / 2)])
        cams.append(CameraPose.from_center(gimbal_rotation(10.0, -60, 0), C))
    return cams


def local_model(cams, X, s=0.1):
    sim = Similarity(s, Rotation.from_euler("z", 40, degrees=True).as_matrix(), np.array([5.0, -3.0, 1.0]))
    traj = Trajectory(np.arange(len(cams)), np.arange(len(cams)) * 0.5, tuple(sim.apply_pose(p) for p in cams),
                      np.ones(len(cams), bool), K, "local", False, "scipy-ba", False)
    return traj, sim


def noisy_gps(cams, sigma_h, sigma_v, seed=0):
    rng = np.random.default_rng(seed)
    C = np.array([p.center for p in cams])
    return C + rng.normal(0, [sigma_h, sigma_h, sigma_v], C.shape), np.tile([sigma_h, sigma_h, sigma_v], (len(C), 1))


def ground_error(reg, X):
    return float(np.max(np.linalg.norm(reg.points - X, axis=1)))


X = np.array([[20.0, 5.0, 0.0], [60.0, -10.0, 3.0], [110.0, 20.0, 1.0]])


def test_barometer_tightens_tilt_and_height():
    cams = level_flight()
    traj, sim = local_model(cams, X)
    gps, sig = noisy_gps(cams, 0.5, 2.0)  # vertical GPS is the weak axis
    rng = np.random.default_rng(3)
    baro = np.array([p.center[2] for p in cams]) - 250.0 + rng.normal(0, 0.1, N)  # take-off offset unknown
    no = georeference(traj, sim.apply_points(X), gps, sig, FRAME)
    yes = georeference(traj, sim.apply_points(X), gps, sig, FRAME, cfg=GeorefConfig(baro_sigma_m=0.1),
                       baro_alt_m=baro)
    assert "barometer" in yes.anchors
    assert np.linalg.norm(yes.rotation_std_deg[:2]) < 0.5 * np.linalg.norm(no.rotation_std_deg[:2])
    assert ground_error(yes, X) < ground_error(no, X)
    assert yes.baro_rms_m < 0.2


def test_known_dimension_fixes_scale_on_a_short_noisy_flight():
    cams = level_flight(n=6, spacing=5.0, curve=6.0)  # ~25 m of track
    traj, sim = local_model(cams, X)
    gps, sig = noisy_gps(cams, 3.0, 5.0, seed=4)  # consumer GPS on a short hop
    a, b = np.array([20.0, 5.0, 0.0]), np.array([60.0, -10.0, 3.0])
    true_len = float(np.linalg.norm(a - b))
    known = [KnownDistance(sim.apply_points(a[None])[0], sim.apply_points(b[None])[0], true_len, 0.02, "wall")]
    no = georeference(traj, sim.apply_points(X), gps, sig, FRAME)
    yes = georeference(traj, sim.apply_points(X), gps, sig, FRAME, known=known)
    assert no.scale_std_rel > 0.02  # GPS alone cannot pin the scale down
    assert yes.scale_std_rel < 0.002
    assert yes.transform.s == pytest.approx(10.0, rel=0.002)
    assert abs(yes.known_residuals_m[0]) < 0.02
    assert "known_dimensions" in yes.anchors and not any("wall" in w for w in yes.warnings)


def test_inconsistent_known_dimension_is_reported():
    cams = level_flight()
    traj, sim = local_model(cams, X)
    gps, sig = noisy_gps(cams, 0.3, 0.5)
    a, b = sim.apply_points(X[:2])
    wrong = [KnownDistance(a, b, 60.0, 0.02, "roof edge")]  # true length is ~43 m
    reg = georeference(traj, sim.apply_points(X), gps, sig, FRAME, known=wrong)
    assert any("'roof edge'" in w and "should be 60.00 m" in w for w in reg.warnings)


def test_barometer_length_is_checked():
    cams = level_flight()
    traj, sim = local_model(cams, X)
    gps, sig = noisy_gps(cams, 0.5, 1.0)
    with pytest.raises(Exception, match="barometer"):
        georeference(traj, sim.apply_points(X), gps, sig, FRAME, baro_alt_m=np.zeros(3))
