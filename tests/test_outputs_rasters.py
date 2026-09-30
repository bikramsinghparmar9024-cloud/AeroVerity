import numpy as np
import pytest
import tifffile

from aerometric.config import RasterConfig
from aerometric.depth.mono import AlignedDepth
from aerometric.geo import LocalFrame
from aerometric.georef.dem import load_dem, read_geotiff
from aerometric.outputs.mesh_io import SurfaceAttributes
from aerometric.outputs.rasters import orthophoto, rasterize_surface, write_surface_rasters
from aerometric.pose.types import CameraPose, Trajectory
from aerometric.segmentation import Label
from aerometric.types import CameraIntrinsics

FRAME = LocalFrame(28.6, 77.2, 200.0)
K = CameraIntrinsics(640, 480, 400, 400, 320, 240, source="provided")


def height(e, n):
    return np.where((np.abs(e - 10) < 8) & (np.abs(n + 5) < 8), 12.0, 0.5 * np.sin(e / 6))


def surface(step=0.5):
    xs, ys = np.arange(-30, 30.01, step), np.arange(-25, 25.01, step)
    X, Y = np.meshgrid(xs, ys)
    V = np.column_stack([X.ravel(), Y.ravel(), height(X.ravel(), Y.ravel())])
    nx = len(xs)
    F = np.array([[j * nx + i, j * nx + i + 1, (j + 1) * nx + i] for j in range(len(ys) - 1) for i in range(nx - 1)]
                 + [[j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i] for j in range(len(ys) - 1)
                    for i in range(nx - 1)])
    roof = V[:, 2] > 6
    attrs = SurfaceAttributes(trust=np.where(roof, 0, 1).astype(np.uint8),
                              sigma_m=np.where(roof, 0.1, 0.5),
                              labels=np.where(roof, Label.BUILDING, Label.ROAD).astype(np.uint8))
    return V, F, attrs


def test_dsm_and_layers_follow_the_surface():
    V, F, attrs = surface()
    r = rasterize_surface(V, F, attrs, RasterConfig(cell_m=1.0))
    E, N = r.grid.centres()
    inside = np.isfinite(r.dsm)
    assert inside.mean() > 0.95
    roof_core = (np.abs(E - 10) < 6) & (np.abs(N + 5) < 6)
    assert np.nanmedian(np.abs(r.dsm[roof_core] - 12.0)) < 1e-6
    ground = (np.abs(E + 20) < 5) & inside
    assert np.max(np.abs(r.dsm[ground] - height(E[ground], N[ground]))) < 0.2
    assert (r.trust[roof_core] == 0).all() and (r.classes[roof_core] == Label.BUILDING).all()
    assert np.nanmax(r.sigma_m[roof_core]) == pytest.approx(0.1)


def test_small_holes_filled_outside_left_empty():
    V, F, attrs = surface()
    # punch a ~2 m hole: move its vertices far away so they land outside the grid window
    hole = (np.abs(V[:, 0] + 15) < 1.2) & (np.abs(V[:, 1]) < 1.2)
    F2 = F[~hole[F].any(1)]
    V2 = V.copy()
    V2[hole] = V[~hole][0]  # collapse them onto an existing vertex; no faces use them any more
    r = rasterize_surface(V2, F2, attrs, RasterConfig(cell_m=1.0, fill_holes_px=2))
    assert r.filled.any()
    E, N = r.grid.centres()
    near_hole = (np.abs(E + 15) < 1) & (np.abs(N) < 1)
    assert np.isfinite(r.dsm[near_hole]).all()
    E, N = r.grid.centres()
    assert not np.isfinite(r.dsm[(E > 31) | (N < -26)]).any()


def test_geotiffs_round_trip_and_open_as_dem(tmp_path):
    V, F, attrs = surface()
    r = rasterize_surface(V, F, attrs, RasterConfig(cell_m=1.0))
    files = write_surface_rasters(tmp_path, r, FRAME)
    assert set(files) == {"dsm", "trust", "class", "sigma"}
    dem = load_dem(files["dsm"])  # our own DEM reader treats the DSM as an elevation model
    pts = np.array([[10.0, -5.0], [-20.0, 3.0]])
    assert dem.height_at_enu(FRAME, pts) == pytest.approx(height(pts[:, 0], pts[:, 1]), abs=0.3)
    arr, lat0, lon0, dlat, dlon, nodata = read_geotiff(files["trust"])
    assert arr.dtype == np.uint8 and nodata == 255
    assert dlat == pytest.approx(np.degrees(1.0 / 6_378_137.0))
    with tifffile.TiffFile(files["dsm"]) as t:
        keys = t.pages[0].tags[34735].value
        assert 4326 in keys and "surface model" in t.pages[0].description


def test_orthophoto_picks_colour_from_a_view_that_sees_the_ground(tmp_path):
    V, F, attrs = surface(step=1.0)
    r = rasterize_surface(V, F, attrs, RasterConfig(cell_m=1.0))
    nadir = np.diag([1.0, -1.0, -1.0])
    poses = [CameraPose.from_center(nadir, np.array([x, 0.0, 60.0])) for x in (-15.0, 15.0)]
    traj = Trajectory(np.arange(2), np.arange(2.0), tuple(poses), np.ones(2, bool), K, "enu", True, "x", False)

    def colour_of(e, n):  # a smooth ground pattern the "camera" photographs
        return np.stack([128 + 100 * np.sin(e / 7), 128 + 100 * np.cos(n / 9), np.full_like(e, 60)], -1)

    def render(k):
        p = poses[k]
        v, u = np.mgrid[0:K.height, 0:K.width].astype(float)
        rays = np.column_stack([(u.ravel() - K.cx) / K.fx, (v.ravel() - K.cy) / K.fy, np.ones(u.size)]) @ p.R_cw
        t = -p.center[2] / rays[:, 2]  # colour of the flat ground point hit (roof shows ground colour below)
        P = p.center + rays * t[:, None]
        rgb = colour_of(P[:, 0], P[:, 1]).reshape(K.height, K.width, 3)
        return np.clip(rgb[..., ::-1], 0, 255).astype(np.uint8)

    ortho, warnings = orthophoto(r, traj, render)
    assert warnings  # no depth maps given
    E, N = r.grid.centres()
    ground = (r.dsm < 1.0) & (ortho[..., 3] == 255)
    expected = colour_of(E[ground], N[ground])
    got = ortho[ground][:, :3].astype(float)
    assert np.median(np.abs(got - expected)) < 3
    files = write_surface_rasters(tmp_path, r, FRAME, ortho)
    back = tifffile.imread(files["ortho"])
    assert back.shape == ortho.shape and np.array_equal(back, ortho)
