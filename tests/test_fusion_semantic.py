import numpy as np

from dronerecon.depth.mono import AlignedDepth
from dronerecon.fusion.semantic import NO_LABEL, vote_labels
from dronerecon.pose.types import CameraPose, Trajectory
from dronerecon.segmentation import Label, SegmentationResult
from dronerecon.types import CameraIntrinsics

W, H, F = 640, 480, 400.0
K = CameraIntrinsics(W, H, F, F, W / 2, H / 2, source="provided")
NADIR = np.diag([1.0, -1.0, -1.0])
BOX = (0.0, 0.0, 10.0, 15.0)  # centre e, centre n, half-size, height


def poses(n=5):
    return [CameraPose.from_center(NADIR, np.array([-40.0 + 20 * i, 0.0, 80.0])) for i in range(n)]


def traj(ps):
    return Trajectory(np.arange(len(ps)), np.arange(len(ps)) * 1.0, tuple(ps), np.ones(len(ps), bool), K,
                      "enu", True, "scipy-ba", False)


def cast(pose, s=1.0):
    """Per-pixel hit point (world) and truth label; resolution W*s x H*s."""
    w, h = int(W * s), int(H * s)
    v, u = np.mgrid[0:h, 0:w].astype(float) / s
    rays = np.column_stack([(u.ravel() - K.cx) / F, (v.ravel() - K.cy) / F, np.ones(u.size)]) @ pose.R_cw
    C = pose.center
    t = -C[2] / rays[:, 2]
    tt = (BOX[3] - C[2]) / rays[:, 2]
    top = C + rays * tt[:, None]
    roof = (np.abs(top[:, 0] - BOX[0]) < BOX[2]) & (np.abs(top[:, 1] - BOX[1]) < BOX[2])
    t = np.where(roof, tt, t)
    P = C + rays * t[:, None]
    lab = np.where(roof, Label.BUILDING, np.where(P[:, 1] < -15, Label.ROAD, Label.VEGETATION))
    depth = (t * np.column_stack([(u.ravel() - K.cx) / F, (v.ravel() - K.cy) / F, np.ones(u.size)])[:, 2])
    return P, lab.reshape(h, w).astype(np.uint8), depth.reshape(h, w).astype(np.float32)


def segs_and_depths(ps, mock=False):
    segs, depths = [], []
    for p in ps:
        _, lab, _ = cast(p)
        segs.append(SegmentationResult(lab, np.full(lab.shape, 0.9, np.float32), "test", mock))
        _, _, d = cast(p, s=0.5)
        depths.append(AlignedDepth(d, 2.0, 1.0, 0.0, "depth", 0.0, 100, True, "test", False))
    return segs, depths


def points():
    ground = np.array([[-30.0, -25.0, 0.0], [30.0, 10.0, 0.0], [-25.0, 20.0, 0.0]])
    roof = np.array([[2.0, 3.0, 15.0], [-5.0, -4.0, 15.0]])
    under = np.array([[1.0, 1.0, 0.0]])  # ground hidden beneath the roof
    return np.concatenate([ground, roof, under])


def test_votes_give_true_classes_with_occlusion_check():
    ps = poses()
    segs, depths = segs_and_depths(ps)
    res = vote_labels(points(), traj(ps), segs, depths)
    assert res.labels[:5].tolist() == [Label.ROAD, Label.VEGETATION, Label.VEGETATION, Label.BUILDING, Label.BUILDING]
    assert res.labels[5] == NO_LABEL and res.votes[5] == 0  # hidden under the roof
    assert res.occlusion_checked and not res.is_mock
    assert np.all(res.confidence[:5] > 0.99)


def test_without_depth_hidden_points_are_mislabelled_and_we_say_so():
    ps = poses()
    segs, _ = segs_and_depths(ps)
    res = vote_labels(points(), traj(ps), segs)
    assert res.labels[5] == Label.BUILDING
    assert any("no depth maps" in w for w in res.warnings)


def test_majority_beats_one_wrong_frame_and_confidence_reflects_it():
    ps = poses()
    segs, depths = segs_and_depths(ps)
    wrong = segs[2]
    segs[2] = SegmentationResult(np.full_like(wrong.labels, Label.WATER), wrong.confidence, "test", False)
    res = vote_labels(points(), traj(ps), segs, depths)
    assert res.labels[3] == Label.BUILDING
    assert 0.5 < res.confidence[3] < 1.0


def test_vehicle_points_are_flagged_dynamic():
    ps = poses()
    segs, depths = segs_and_depths(ps)
    for i in range(len(segs)):
        lab = segs[i].labels.copy()
        lab[:] = np.where(lab == Label.ROAD, Label.VEHICLE, lab)  # a car parked on the road in every frame
        segs[i] = SegmentationResult(lab, segs[i].confidence, "test", False)
    res = vote_labels(points(), traj(ps), segs, depths)
    assert res.dynamic.tolist() == [True, False, False, False, False, False]
    assert res.class_counts()["vehicle"] == 1


def test_mock_segmenter_is_reported():
    ps = poses()
    segs, depths = segs_and_depths(ps, mock=True)
    res = vote_labels(points(), traj(ps), segs, depths)
    assert res.is_mock and any("placeholder segmenter" in w for w in res.warnings)
