import numpy as np
import pytest

from dronerecon.config import FusionConfig, UncertaintyConfig
from dronerecon.depth.mono import frame_observations, make_depth_model
from dronerecon.pose.tracks import build_tracks, triangulate
from dronerecon.pose.geometry import normalized_points
from dronerecon.uncertainty.bootstrap import (
    draw_runs,
    pose_run,
    run_bootstrap,
    summarise,
    surface_offsets,
    surface_run,
)
from test_pose_classical import K, N, gps_for, initial_from, observe, truth


def small_scene(n_points=800):
    cams, X = truth(curved=True)
    X[:, 0] -= 30
    X = X[np.random.default_rng(7).choice(len(X), n_points, replace=False)]
    kps, pairs = observe(cams, X, noise=0.3)
    return cams, kps, pairs, build_tracks([len(k) for k in kps], pairs)


def test_draws_are_reproducible_and_sized():
    gps, sig = np.zeros((10, 3)), np.ones((10, 3))
    a = draw_runs(10, gps, sig, UncertaintyConfig(n_runs=4), seed=5)
    b = draw_runs(10, gps, sig, UncertaintyConfig(n_runs=4), seed=5)
    assert all(np.array_equal(x.keep, y.keep) and np.array_equal(x.gps, y.gps) for x, y in zip(a, b))
    assert all(d.keep.sum() == 8 for d in a)
    assert 0.5 < np.std(np.concatenate([d.gps for d in a])) < 1.5


def test_summary_recovers_known_spread():
    rng = np.random.default_rng(0)
    true_std = np.linspace(0.1, 2.0, 50)
    samples = rng.normal(0, 1, (400, 50)) * true_std
    samples[:, :5][rng.random((400, 5)) < 0.999] = np.nan  # almost never measured
    with np.errstate(all="ignore"):
        st = summarise(samples, UncertaintyConfig(min_valid_runs=3))
    assert st.std[5:] == pytest.approx(true_std[5:], rel=0.15)
    assert np.isnan(st.std[:5]).all()


def test_surface_offsets_signed_along_normal():
    q = np.array([[0.0, 0, 0], [5, 0, 0], [50, 0, 0]])
    nrm = np.array([[0, 0, 1.0]] * 3)
    run = np.array([[0.0, 0.0, 0.3], [5.0, 0.1, -0.2]])
    off = surface_offsets(q, nrm, run, max_gap=1.0)
    assert off[:2] == pytest.approx([0.3, -0.2]) and np.isnan(off[2])


def test_pose_bootstrap_spread_tracks_gps_noise():
    cams, kps, pairs, tracks = small_scene()
    init = initial_from(cams)

    def spread(gps_sigma):
        gps, sig = gps_for(cams, sigma=gps_sigma)
        cfg = UncertaintyConfig(n_runs=4)
        st = run_bootstrap(draw_runs(N, gps, sig, cfg, seed=1),
                           lambda d: pose_run(d, kps, pairs, init, sig, tracks).points, cfg)
        return np.nanmedian(st.std)

    tight, loose = spread(0.2), spread(2.0)
    assert loose > 3 * tight  # position uncertainty follows the GPS it rests on


def test_surface_bootstrap_runs_end_to_end():
    cams, kps, pairs, tracks = small_scene()
    init = initial_from(cams, rot_err_deg=0.0, pos_err_m=0.0)
    gps, sig = gps_for(cams, sigma=1.0)
    ucfg = UncertaintyConfig(n_runs=3, min_valid_runs=2, max_gap_m=3.0)

    # depth predictions from the stand-in, built once from the full-run geometry
    norm = [normalized_points(k, K) for k in kps]
    pts = np.array([triangulate(init.poses, np.array([norm[c][k] for c, k in t]), t[:, 0]) for t in tracks])
    model = make_depth_model()
    small = np.zeros((135, 240, 3), np.uint8)
    preds = [model.predict(small, frame_observations(init, pts, tracks, kps, i), K.width) for i in range(N)]
    base_verts = surface_run(pose_run(draw_runs(N, gps, sig, UncertaintyConfig(n_runs=1, keep_fraction=1.0))[0],
                                      kps, pairs, init, sig, tracks),
                             init, preds, tracks, kps, K.width, fusion_cfg=FusionConfig(voxel_m=2.0))
    query = base_verts[::max(1, len(base_verts) // 200)]
    normals = np.tile([0, 0, 1.0], (len(query), 1))

    def run_one(d):
        pr = pose_run(d, kps, pairs, init, sig, tracks)
        verts = surface_run(pr, init, preds, tracks, kps, K.width, fusion_cfg=FusionConfig(voxel_m=2.0))
        return surface_offsets(query, normals, verts, ucfg.max_gap_m)

    st = run_bootstrap(draw_runs(N, gps, sig, ucfg, seed=2), run_one, ucfg)
    assert st.n_runs == 3
    assert np.isfinite(st.std).mean() > 0.5
    assert np.nanmedian(st.std) < 2.0
