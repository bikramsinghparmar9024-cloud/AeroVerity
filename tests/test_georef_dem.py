import numpy as np
import pytest
import tifffile
from scipy.spatial.transform import Rotation

from aerometric.geo import LocalFrame
from aerometric.georef.dem import ElevationModel, load_dem, register_to_dem
from aerometric.georef.similarity import GeoRegistration
from aerometric.pose.types import CameraPose, Similarity, Trajectory
from aerometric.types import CameraIntrinsics, DataError

FRAME = LocalFrame(28.6, 77.2, 0.0)
K = CameraIntrinsics(1920, 1080, 1400, 1400, 960, 540, source="provided")
M_PER_DEG_LAT = 111_320.0


def terrain(e, n, relief=1.0):
    return relief * (12 * np.sin(e / 40) + 9 * np.cos(n / 35) + 0.02 * e)


def make_dem(relief=1.0, cell_deg=1e-4):
    lat = np.arange(28.6 + 0.004, 28.6 - 0.004, -cell_deg)
    lon = np.arange(77.2 - 0.004, 77.2 + 0.004, cell_deg)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    enu = FRAME.to_enu(LAT.ravel(), LON.ravel(), np.zeros(LAT.size))
    H = terrain(enu[:, 0], enu[:, 1], relief).reshape(LAT.shape)
    return ElevationModel(H, lat[0], lon[0], cell_deg, cell_deg)


def ground_points(n=3000, relief=1.0, buildings=0.0, seed=0):
    rng = np.random.default_rng(seed)
    e, nn = rng.uniform(-150, 150, n), rng.uniform(-150, 150, n)
    z = terrain(e, nn, relief)
    roof = rng.random(n) < buildings
    z[roof] += rng.uniform(6, 15, roof.sum())
    return np.column_stack([e, nn, z])


def registration(points, error: Similarity):
    poses = tuple(CameraPose.from_center(np.diag([1.0, -1, -1]), np.array([i * 20.0 - 100, 0, 120])) for i in range(10))
    traj = Trajectory(np.arange(10), np.arange(10.0), poses, np.ones(10, bool), K, "enu", True, "scipy-ba", False)
    moved = traj.transformed(error, "enu", True)
    return GeoRegistration(error, moved, error.apply_points(points), FRAME, 0.8, np.zeros(10, bool), float("nan"),
                           float("nan"), np.empty(0), np.array([0.3, 0.3, 0.3]), 0.01, True, ("gps",))


def error_transform(dz=7.0, tilt_deg=0.4, shift=(4.0, -3.0)):
    R = Rotation.from_rotvec(np.radians(tilt_deg) * np.array([0.6, 0.8, 0])).as_matrix()
    return Similarity(1.0, R, np.array([shift[0], shift[1], dz]))


def test_dem_lookup_matches_terrain():
    dem = make_dem()
    xy = np.array([[10.0, 20.0], [-80.0, 55.0]])
    assert dem.height_at_enu(FRAME, xy) == pytest.approx(terrain(xy[:, 0], xy[:, 1]), abs=0.3)
    assert np.isnan(dem.height_at_enu(FRAME, np.array([[5000.0, 0.0]])))[0]


def test_registration_recovers_height_tilt_and_shift_on_hilly_ground():
    X = ground_points(buildings=0.2)
    reg = registration(X, error_transform())
    out = register_to_dem(reg, make_dem())
    assert out.shift_estimated
    ground = ~(X[:, 2] - terrain(X[:, 0], X[:, 1]) > 3)
    err = np.linalg.norm(out.registration.points[ground] - X[ground], axis=1)
    assert np.median(err) < 1.0
    assert out.height_rms_m < 1.0
    assert out.registration.anchors[-1] == "dem"
    assert not out.warnings  # 7 m is within ordinary GPS height error


def test_flat_ground_adjusts_height_only_and_says_so():
    X = ground_points(relief=0.05)
    # -40 m: typical of GPS ellipsoidal height vs a mean-sea-level DEM (geoid undulation)
    reg = registration(X, error_transform(dz=-40.0, tilt_deg=0.0, shift=(6.0, 0.0)))
    out = register_to_dem(reg, make_dem(relief=0.05))
    assert not out.shift_estimated
    assert out.correction.t[2] == pytest.approx(40.0, abs=0.3)
    assert any("too flat" in w for w in out.warnings)
    assert any("vertical datums" in w for w in out.warnings)


def test_esri_ascii_and_geotiff_loaders(tmp_path):
    dem = make_dem(cell_deg=5e-4)
    h, w = dem.heights.shape
    body = "\n".join(" ".join(f"{v:.3f}" for v in row) for row in dem.heights)
    asc = tmp_path / "dem.asc"
    asc.write_text(f"ncols {w}\nnrows {h}\nxllcenter {dem.lon_left}\n"
                   f"yllcenter {dem.lat_top - (h - 1) * dem.dlat}\ncellsize {dem.dlat}\nNODATA_value -9999\n" + body)
    tif = tmp_path / "dem.tif"
    tifffile.imwrite(tif, dem.heights.astype(np.float32), extratags=[
        (33550, "d", 3, (dem.dlon, dem.dlat, 0.0), True),
        (33922, "d", 6, (0.0, 0.0, 0.0, dem.lon_left - dem.dlon / 2, dem.lat_top + dem.dlat / 2, 0.0), True),
    ])
    xy = np.array([[30.0, -40.0]])
    for path in (asc, tif):
        assert load_dem(path).height_at_enu(FRAME, xy) == pytest.approx(dem.height_at_enu(FRAME, xy), abs=0.01)


def test_projected_dem_is_refused(tmp_path):
    asc = tmp_path / "utm.asc"
    asc.write_text("ncols 2\nnrows 2\nxllcorner 712000\nyllcorner 3165000\ncellsize 30\n1 2\n3 4\n")
    with pytest.raises(DataError, match="reproject"):
        load_dem(asc)


def test_dem_elsewhere_is_refused():
    X = ground_points() + [5000.0, 0, 0]
    with pytest.raises(DataError, match="fall on the elevation model"):
        register_to_dem(registration(X, Similarity(1.0, np.eye(3), np.zeros(3))), make_dem())
