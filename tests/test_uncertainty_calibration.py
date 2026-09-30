import numpy as np
import pytest

from dronerecon.pose.types import CameraPose, Trajectory
from dronerecon.types import CameraIntrinsics
from dronerecon.uncertainty.bootstrap import RunDraw
from dronerecon.uncertainty.calibration import calibrate, held_out_errors, reliability_svg

K = CameraIntrinsics(1000, 800, 800, 800, 500, 400, source="provided")


def synthetic(true_scale, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(0.05, 1.0, n)
    e2d = rng.normal(0, 1, (n, 2)) * (sigma * true_scale)[:, None]
    return sigma, np.linalg.norm(e2d, axis=1)


@pytest.mark.parametrize("true_scale", [0.4, 1.0, 2.5])
def test_scale_is_recovered_and_coverage_fixed(true_scale):
    sigma, err = synthetic(true_scale)
    cal = calibrate(sigma, err)
    assert cal.scale == pytest.approx(true_scale, rel=0.05)
    assert np.abs(cal.coverage_after - cal.levels).max() < 0.03
    if true_scale != 1.0:
        assert np.abs(cal.coverage_before - cal.levels).max() > 0.2
    assert cal.bin_rms_error == pytest.approx(cal.bin_sigma, rel=0.15)  # sharp and reliable per bin
    assert cal.trusted


def test_warnings_for_big_miscalibration_and_small_samples():
    sigma, err = synthetic(3.0, n=30)
    cal = calibrate(sigma, err)
    assert not cal.trusted
    assert any("understated" in w for w in cal.warnings) and any("rough guide" in w for w in cal.warnings)
    sigma, err = synthetic(0.3)
    assert any("overstated" in w for w in calibrate(sigma, err).warnings)


def test_held_out_errors_measure_the_miss_in_metres():
    nadir = np.diag([1.0, -1.0, -1.0])
    poses = tuple(CameraPose.from_center(nadir, np.array([i * 10.0, 0, 100.0])) for i in range(3))
    base = Trajectory(np.arange(3), np.arange(3.0), poses, np.ones(3, bool), K, "enu", True, "scipy-ba", False)
    X = np.array([[5.0, 2.0, 0.0], [12.0, -3.0, 0.0]])

    def project(c, x):
        Xc = poses[c].R_cw @ x + poses[c].t_cw
        return np.array([K.fx * Xc[0] / Xc[2] + K.cx, K.fy * Xc[1] / Xc[2] + K.cy])

    keypoints = [np.array([project(c, X[0]), project(c, X[1])]) for c in range(3)]
    tracks = [np.array([[0, 0], [1, 0], [2, 0]]), np.array([[0, 1], [1, 1], [2, 1]])]
    run_pts = X + [[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]  # the run got point 0 wrong by 30 cm east
    draw = RunDraw(np.array([True, False, True]), np.zeros((3, 3)))  # keyframe 1 was held out
    ho = held_out_errors([run_pts], [draw], base, tracks, keypoints)
    assert ho.point_ids.tolist() == [0, 1]
    assert ho.error_m == pytest.approx([0.3, 0.0], abs=1e-6)
    assert ho.error_px[0] == pytest.approx(0.3 * K.fx / 100.0)


def test_reliability_svg_is_well_formed():
    import xml.etree.ElementTree as ET

    sigma, err = synthetic(1.7)
    svg = reliability_svg(calibrate(sigma, err))
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert len(root.findall("{http://www.w3.org/2000/svg}polyline")) == 2
    assert "calibrated (×1.6" in svg or "calibrated (×1.7" in svg
