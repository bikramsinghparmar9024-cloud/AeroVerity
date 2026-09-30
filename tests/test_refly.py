import numpy as np
import pytest

from dronerecon.config import ReflyConfig
from dronerecon.geo import LocalFrame
from dronerecon.pose.types import CameraPose, Trajectory
from dronerecon.quality.coverage import WeakRegion
from dronerecon.refly import advise
from dronerecon.trust import TrustLabel, assess
from dronerecon.types import CameraIntrinsics

K = CameraIntrinsics(800, 600, 500, 500, 400, 300, source="provided")
NADIR = np.diag([1.0, -1.0, -1.0])


def flight():
    poses = tuple(CameraPose.from_center(NADIR, np.array([x, 0.0, 80.0])) for x in np.arange(-40, 41, 10.0))
    return Trajectory(np.arange(len(poses)), np.arange(len(poses)) * 1.0, poses, np.ones(len(poses), bool), K,
                      "enu", True, "scipy-ba", False)


def grid_mesh(xs, ys, z_fn, normal):
    X, Y = np.meshgrid(xs, ys)
    V = np.column_stack([X.ravel(), Y.ravel(), z_fn(X.ravel(), Y.ravel())])
    nx = len(xs)
    F = []
    for j in range(len(ys) - 1):
        for i in range(nx - 1):
            a = j * nx + i
            F += [[a, a + 1, a + nx], [a + 1, a + nx + 1, a + nx]]
    return V, np.array(F), np.tile(normal, (len(V), 1))


def scene():
    """Ground (well measured) plus an east-facing wall at x=20 (seen only edge-on from above)."""
    gv, gf, gn = grid_mesh(np.arange(-40, 41, 2.0), np.arange(-30, 31, 2.0), lambda x, y: np.zeros_like(x), [0, 0, 1.0])
    ys, zs = np.arange(-10, 11, 1.0), np.arange(0, 15.1, 1.0)
    Y, Z = np.meshgrid(ys, zs)
    wv = np.column_stack([np.full(Y.size, 20.0), Y.ravel(), Z.ravel()])
    nx = len(ys)
    wf = np.array([[j * nx + i, j * nx + i + 1, (j + 1) * nx + i] for j in range(len(zs) - 1) for i in range(nx - 1)]
                  + [[j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i] for j in range(len(zs) - 1)
                     for i in range(nx - 1)])
    V = np.concatenate([gv, wv])
    F = np.concatenate([gf, wf + len(gv)])
    N = np.concatenate([gn, np.tile([1.0, 0, 0], (len(wv), 1))])
    sigma = np.concatenate([np.full(len(gv), 0.1), np.full(len(wv), 3.0)])  # wall: huge uncertainty
    trust = assess(sigma, np.ones(len(V)), np.full(len(V), 5.0))
    return V, F, N, trust, len(gv)


def test_wall_seen_edge_on_gets_facing_viewpoints():
    V, F, N, trust, n_ground = scene()
    adv = advise(V, F, N, trust, flight(), frame=LocalFrame(28.6, 77.2, 200.0))
    assert len(adv) == 1
    a = adv[0]
    assert a.center_enu == pytest.approx([20, 0, 7.5], abs=0.6)
    assert a.area_m2 == pytest.approx(20 * 15, rel=0.1)
    assert a.reason == "grazing angle" and a.existing_views >= 3
    for v in a.viewpoints:
        assert v.position_enu[0] > 30  # east of the wall, facing it
        assert 225 < v.heading_deg < 315  # camera looks roughly west
        assert v.position_enu[2] >= 30.0  # minimum height above the local ground
        assert np.isfinite(v.lat) and v.alt_m > 200
    d = a.viewpoints[0].position_enu - a.viewpoints[1].position_enu
    assert np.linalg.norm(d) > 3  # two distinct shots with parallax
    assert "from the east" in a.message and "Unknown" in a.message


def test_out_of_view_region_and_moving_objects():
    V, F, N, trust, n_ground = scene()
    far = np.flatnonzero((V[:, 0] < -30) & (V[:, 2] == 0))  # beyond the first camera's footprint edge
    labels = trust.labels.copy()
    labels[far] = TrustLabel.UNKNOWN
    labels[n_ground:] = TrustLabel.MEASURED
    from dataclasses import replace
    t2 = replace(trust, labels=labels)
    adv = advise(V, F, N, t2, flight(), cfg=ReflyConfig(min_area_m2=10))
    assert any(a.reason in ("never in view", "seen by too few frames") for a in adv)
    dyn = np.zeros(len(V), bool)
    dyn[far] = True  # if those were cars, re-flying would not help
    assert advise(V, F, N, t2, flight(), dynamic=dyn, cfg=ReflyConfig(min_area_m2=10)) == []


def test_coverage_gaps_become_advice_and_ranking():
    V, F, N, trust, _ = scene()
    gaps = [WeakRegion((-70.0, 0.0), 900.0, 56), WeakRegion((70.0, 0.0), 10.0, 1)]
    adv = advise(V, F, N, trust, flight(), gaps=gaps, ground_z=0.0)
    reasons = [a.reason for a in adv]
    assert reasons[0] == "coverage gap" and "grazing angle" in reasons  # 900 m² gap outranks the wall
    assert sum(r == "coverage gap" for r in reasons) == 1  # the tiny gap is ignored
    gap = adv[0]
    assert all(v.gimbal_pitch_deg < -60 for v in gap.viewpoints)  # ground: look steeply down


def test_nothing_to_advise_when_everything_is_measured():
    V, F, N, trust, _ = scene()
    from dataclasses import replace
    all_ok = replace(trust, labels=np.zeros(len(V), np.uint8))
    assert advise(V, F, N, all_ok, flight()) == []
