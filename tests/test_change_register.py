import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from aerometric.change.register import apply, icp, ortho_prior, register_flights, to_frame
from aerometric.geo import LocalFrame
from aerometric.pose.types import Similarity

FRAME_A = LocalFrame(28.6000, 77.2000, 210.0)
FRAME_B = LocalFrame(28.6004, 77.1997, 205.0)  # flight B took off elsewhere


def terrain(e, n):
    return 6 * np.sin(e / 23) * np.cos(n / 31) + 0.05 * e + 3 * np.exp(-((e - 40) ** 2 + (n + 30) ** 2) / 300)


def sample(n_pts, seed, new_building=False):
    rng = np.random.default_rng(seed)
    e, n = rng.uniform(-100, 100, n_pts), rng.uniform(-100, 100, n_pts)
    z = terrain(e, n)
    if new_building:  # 20 x 20 m, 10 m tall block that did not exist in flight A
        inside = (np.abs(e + 30) < 10) & (np.abs(n - 20) < 10)
        z = np.where(inside, z + 10.0, z)
    return np.column_stack([e, n, z])


def gps_error(yaw_deg=1.0, shift=(3.0, -2.0, 1.0)):
    R = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    return Similarity(1.0, R, np.asarray(shift))


def flight_b(err: Similarity, new_building=True):
    """Flight B's points in its own frame, carrying a GPS placement error."""
    truth_in_a = sample(6000, seed=2, new_building=new_building)
    wrong_in_a = err.apply_points(truth_in_a)
    return to_frame(wrong_in_a, FRAME_A, FRAME_B), truth_in_a


def test_frames_round_trip():
    P = sample(50, 0)
    assert to_frame(to_frame(P, FRAME_A, FRAME_B), FRAME_B, FRAME_A) == pytest.approx(P, abs=1e-6)


def test_icp_recovers_gps_error_despite_a_new_building():
    A = sample(8000, seed=1)
    err = gps_error()
    B, truth = flight_b(err)
    reg = register_flights(A, FRAME_A, B, FRAME_B)
    aligned = apply(reg, B, FRAME_B, FRAME_A)
    unchanged = ~((np.abs(truth[:, 0] + 30) < 12) & (np.abs(truth[:, 1] - 20) < 12))
    assert np.median(np.linalg.norm(aligned[unchanged] - truth[unchanged], axis=1)) < 0.15
    assert reg.converged and reg.overlap > 0.9 and reg.prior == "gps"
    assert not any("correction" in w for w in reg.warnings)  # 3.6 m / 1 deg is below the note threshold


def test_low_overlap_is_reported():
    A = sample(6000, seed=1)
    A = A[A[:, 0] < -60]  # A only covers a 40 m western strip (20 % of B's area)
    B, _ = flight_b(gps_error(yaw_deg=0.2, shift=(0.5, 0.3, 0.2)), new_building=False)
    reg = register_flights(A, FRAME_A, B, FRAME_B)
    assert reg.overlap < 0.3 and any("overlaps" in w for w in reg.warnings)


def test_icp_needs_points():
    with pytest.raises(Exception, match="at least 10"):
        icp(np.zeros((3, 3)), np.zeros((50, 3)))


def texture(seed=0, size=900):
    rng = np.random.default_rng(seed)
    img = cv2.resize((rng.random((size // 12, size // 12, 3)) * 255).astype(np.uint8), (size, size),
                     interpolation=cv2.INTER_CUBIC)
    for _ in range(500):
        x, y = rng.integers(0, size - 30, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(5, 30)), int(y + rng.integers(5, 30))),
                      tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return img


def test_ortho_prior_gives_metric_shift_and_heading():
    cell = 0.5
    world = texture()
    a = world[100:700, 100:700]
    # flight B's ortho is the same ground, rotated 3 degrees and shifted 12 m east / 8 m south in pixels
    M = cv2.getRotationMatrix2D((400.0, 400.0), 3.0, 1.0)
    M[:, 2] += (-24.0, 16.0)
    b = cv2.warpAffine(world, M, (900, 900))[100:700, 100:700]
    to_rgba = lambda im: np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])  # noqa: E731
    origin_a = (-150.0, 150.0)
    prior = ortho_prior(to_rgba(a), to_rgba(b), cell, origin_a)
    assert prior is not None
    # a B-orthophoto pixel and its true A pixel, both converted to metres
    def en(col, row):
        return np.array([origin_a[0] + col * cell, origin_a[1] - row * cell, 0.0])
    Minv = cv2.invertAffineTransform(M)
    for col, row in [(150, 150), (450, 300), (300, 480)]:
        src = np.array([col + 100, row + 100, 1.0])
        true_world = Minv @ src  # where that B pixel really is in the world texture
        expected = en(true_world[0] - 100, true_world[1] - 100)
        assert prior.apply_points(en(col, row)[None])[0][:2] == pytest.approx(expected[:2], abs=0.5)
