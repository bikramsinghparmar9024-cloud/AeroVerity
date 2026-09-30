import cv2
import numpy as np
import pytest
import tifffile

from dronerecon.config import GeorefConfig
from dronerecon.geo import LocalFrame
from dronerecon.georef.satellite import GeoImage, load_geoimage, register_to_satellite
from dronerecon.georef.similarity import GeoRegistration
from dronerecon.pose.types import CameraPose, Similarity, Trajectory
from dronerecon.types import CameraIntrinsics, DataError

FRAME = LocalFrame(28.6, 77.2, 0.0)
K = CameraIntrinsics(640, 480, 500, 500, 320, 240, source="provided")
RES = 0.5  # satellite metres per pixel
HALF = 300.0  # satellite covers +-300 m around the origin
DEG_LAT = RES / 111_319.49  # degrees of latitude per 0.5 m (matches geo.LocalFrame radius)


def world_texture(seed=0):
    rng = np.random.default_rng(seed)
    n = int(2 * HALF / RES)
    img = cv2.resize((rng.random((n // 16, n // 16, 3)) * 255).astype(np.uint8), (n, n),
                     interpolation=cv2.INTER_CUBIC)
    for _ in range(900):
        x, y = rng.integers(0, n - 40, 2)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(6, 40)), int(y + rng.integers(6, 40))), color, -1)
        else:
            cv2.circle(img, (int(x), int(y)), int(rng.integers(3, 14)), color, -1)
    return img


def geoimage(img):
    dlon = DEG_LAT / np.cos(np.radians(FRAME.lat0))
    n = img.shape[0]
    return GeoImage(img, FRAME.lat0 + (n / 2 - 0.5) * DEG_LAT, FRAME.lon0 - (n / 2 - 0.5) * dlon, DEG_LAT, dlon)


def render(tex_geo: GeoImage, pose: CameraPose, z0=0.0):
    """Photograph the flat textured ground from a camera pose."""
    u, v = np.meshgrid(np.arange(K.width), np.arange(K.height))
    rays = np.column_stack([(u.ravel() - K.cx) / K.fx, (v.ravel() - K.cy) / K.fy, np.ones(u.size)]) @ pose.R_cw
    C = pose.center
    s = (z0 - C[2]) / rays[:, 2]
    P = C + rays * s[:, None]
    lat, lon, _ = FRAME.to_lla(P)
    mx = ((lon - tex_geo.lon_left) / tex_geo.dlon).reshape(K.height, K.width).astype(np.float32)
    my = ((tex_geo.lat_top - lat) / tex_geo.dlat).reshape(K.height, K.width).astype(np.float32)
    return cv2.remap(tex_geo.image, mx, my, cv2.INTER_LINEAR)


def flight(n=6):
    R = np.diag([1.0, -1.0, -1.0])  # nadir, image x = east, image y = south
    return [CameraPose.from_center(R, np.array([-100 + 40.0 * i, 10.0, 110.0])) for i in range(n)]


def registration(true_poses, err_en, gps_rms=5.0):
    err = Similarity(1.0, np.eye(3), np.array([err_en[0], err_en[1], 0.0]))
    traj = Trajectory(np.arange(len(true_poses)), np.arange(len(true_poses)) * 1.0,
                      tuple(err.apply_pose(p) for p in true_poses), np.ones(len(true_poses), bool), K,
                      "enu", True, "scipy-ba", False)
    pts = err.apply_points(np.array([[0.0, 0.0, 0.0], [50.0, 20.0, 0.0]]))
    return GeoRegistration(err, traj, pts, FRAME, gps_rms, np.zeros(len(true_poses), bool), float("nan"), float("nan"),
                           np.empty(0), np.array([0.2, 0.2, 0.2]), 0.01, True, ("gps",))


@pytest.fixture(scope="module")
def scene():
    geo = geoimage(world_texture())
    poses = flight()
    images = [render(geo, p) for p in poses]
    return geo, poses, images


def test_recovers_horizontal_offset(scene):
    geo, poses, images = scene
    reg = registration(poses, (6.0, -4.0))
    out = register_to_satellite(reg, geo, lambda k: images[k], GeorefConfig(sat_sigma_m=0.3))
    assert out.frames_matched >= 4
    assert out.measured_shift_en == pytest.approx([-6.0, 4.0], abs=0.6)
    # GPS says +-5 m, imagery +-0.3 m: almost all of the correction is applied
    fixed = out.registration.trajectory.centers[:, :2]
    assert np.abs(fixed - [p.center[:2] for p in poses]).max() < 0.7
    assert out.registration.anchors[-1] == "satellite" and not out.warnings


def test_blending_respects_uncertainties(scene):
    geo, poses, images = scene
    reg = registration(poses, (6.0, -4.0), gps_rms=0.5)  # RTK-grade position already
    out = register_to_satellite(reg, geo, lambda k: images[k], GeorefConfig(sat_sigma_m=3.0))
    assert np.linalg.norm(out.applied_shift_en) < 0.1 * np.linalg.norm(out.measured_shift_en)
    assert not out.warnings  # 7 m is within 3x the imagery's own 3 m accuracy

    far = registration(poses, (20.0, -15.0), gps_rms=0.5)  # 25 m disagreement
    out = register_to_satellite(far, geo, lambda k: images[k], GeorefConfig(sat_sigma_m=3.0))
    assert out.measured_shift_en == pytest.approx([-20.0, 15.0], abs=1.0)
    assert any("check the imagery" in w for w in out.warnings)


def test_wrong_imagery_changes_nothing(scene):
    _, poses, images = scene
    other = geoimage(world_texture(seed=9))
    reg = registration(poses, (6.0, -4.0))
    out = register_to_satellite(reg, other, lambda k: images[k])
    assert np.all(out.applied_shift_en == 0)
    assert out.registration is reg
    assert any("left unchanged" in w for w in out.warnings)


def test_loaders(tmp_path):
    img = world_texture()[:200, :300]
    geo = geoimage(world_texture())
    png = tmp_path / "sat.png"
    cv2.imwrite(str(png), img)
    png.with_suffix(".pgw").write_text(f"{geo.dlon}\n0\n0\n{-geo.dlat}\n{geo.lon_left}\n{geo.lat_top}\n")
    a = load_geoimage(png)
    assert a.image.shape == img.shape and a.lat_top == pytest.approx(geo.lat_top)

    tif = tmp_path / "sat.tif"
    tifffile.imwrite(tif, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), photometric="rgb", extratags=[
        (33550, "d", 3, (geo.dlon, geo.dlat, 0.0), True),
        (33922, "d", 6, (0.0, 0.0, 0.0, geo.lon_left - geo.dlon / 2, geo.lat_top + geo.dlat / 2, 0.0), True)])
    b = load_geoimage(tif)
    assert np.array_equal(b.image, img) and b.lon_left == pytest.approx(geo.lon_left)

    bare = tmp_path / "plain.jpg"
    cv2.imwrite(str(bare), img)
    with pytest.raises(DataError, match="no georeferencing"):
        load_geoimage(bare)
