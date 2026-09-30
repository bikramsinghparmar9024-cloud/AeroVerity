import struct

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from aerometric.geo import LocalFrame
from aerometric.outputs.points_io import SH_C0, write_las, write_splat_seed
from aerometric.segmentation import Label
from aerometric.types import DataError

FRAME = LocalFrame(28.6, 77.2, 210.0)


def read_las(path):
    b = path.read_bytes()
    h = struct.unpack("<4sHH16sBB32s32sHHHIIBHI5I3d3d6d", b[:227])
    sig, (vmaj, vmin), hsize, off, nvlr, fmt, rlen, n = h[0], h[4:6], h[10], h[11], h[12], h[13], h[14], h[15]
    scale, offset, bounds = np.array(h[21:24]), np.array(h[24:27]), h[27:33]
    vlr_user = b[227 + 2:227 + 18].rstrip(b"\0")
    vlr_rec = struct.unpack("<H", b[227 + 18:227 + 20])[0]
    rec = np.frombuffer(b[off:off + n * rlen], dtype=[
        ("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"), ("intensity", "<u2"), ("flags", "u1"), ("cls", "u1"),
        ("scan", "i1"), ("user", "u1"), ("src", "<u2"), ("r", "<u2"), ("g", "<u2"), ("b", "<u2")])
    xyz = np.column_stack([rec["X"], rec["Y"], rec["Z"]]) * scale + offset
    return dict(sig=sig, version=(vmaj, vmin), hsize=hsize, nvlr=nvlr, fmt=fmt, rlen=rlen, n=n, xyz=xyz, rec=rec,
                bounds=bounds, vlr_user=vlr_user, vlr_rec=vlr_rec, size=len(b), off=off)


def test_las_round_trip(tmp_path):
    P = np.array([[0.0, 0.0, 0.0], [120.5, -40.25, 12.3], [-15.0, 300.0, -2.0]])
    rgb = np.array([[255, 0, 0], [0, 128, 255], [10, 20, 30]], np.uint8)
    labels = np.array([Label.ROAD, Label.BUILDING, 255], np.uint8)
    trust = np.array([0, 1, 2], np.uint8)
    las = read_las(write_las(tmp_path / "cloud.las", P, FRAME, rgb, labels, trust))
    assert las["sig"] == b"LASF" and las["version"] == (1, 2) and las["fmt"] == 2 and las["rlen"] == 26
    assert las["n"] == 3 and las["size"] == las["off"] + 3 * 26
    assert las["vlr_user"] == b"LASF_Projection" and las["vlr_rec"] == 34735
    lat, lon, h = FRAME.to_lla(P)
    assert las["xyz"][:, 0] == pytest.approx(lon, abs=1e-7)
    assert las["xyz"][:, 1] == pytest.approx(lat, abs=1e-7)
    assert las["xyz"][:, 2] == pytest.approx(h, abs=1e-3)
    assert las["rec"]["cls"].tolist() == [2, 6, 1]  # ASPRS ground, building, unclassified
    assert las["rec"]["user"].tolist() == [0, 1, 2]
    assert las["rec"]["r"][0] == 65535 and las["rec"]["b"][1] == 65535
    assert las["bounds"][0] == pytest.approx(lon.max(), abs=1e-9)


def test_laz_and_bad_input_are_refused(tmp_path):
    with pytest.raises(DataError, match="lazrs"):
        write_las(tmp_path / "cloud.laz", np.zeros((1, 3)), FRAME)
    with pytest.raises(DataError, match="rgb has 1 values"):
        write_las(tmp_path / "cloud.las", np.zeros((2, 3)), FRAME, rgb=np.zeros((1, 3), np.uint8))


def read_splat(path):
    b = path.read_bytes()
    head, _, body = b.partition(b"end_header\n")
    names = [l.split()[-1] for l in head.decode().splitlines() if l.startswith("property")]
    return head.decode(), names, np.frombuffer(body, "<f4").reshape(-1, len(names))


def test_splat_seed_layout_and_values(tmp_path):
    xs, ys = np.meshgrid(np.arange(0, 5.0), np.arange(0, 4.0))
    P = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    N = np.tile([0.0, 0.0, 1.0], (len(P), 1))
    N[0] = [1.0, 0.0, 0.0]  # one point on a wall facing east
    N[1] = [0.0, 0.0, -1.0]  # one facing straight down
    rgb = np.tile([200, 100, 50], (len(P), 1)).astype(np.uint8)
    trust = np.zeros(len(P), np.uint8)
    trust[2] = 2
    head, names, a = read_splat(write_splat_seed(tmp_path / "splat.ply", P, N, rgb, trust, FRAME))
    assert "UNTRAINED" in head and names[:3] == ["x", "y", "z"] and names[-1] == "rot_3"
    col = {k: i for i, k in enumerate(names)}
    assert a[:, [col["x"], col["y"], col["z"]]] == pytest.approx(P)
    rgb_back = (a[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]] * SH_C0 + 0.5) * 255
    assert rgb_back[5] == pytest.approx([200, 100, 50], abs=0.01)
    alpha = 1 / (1 + np.exp(-a[:, col["opacity"]]))
    assert alpha[3] == pytest.approx(0.95, abs=1e-4) and alpha[2] == pytest.approx(0.25, abs=1e-4)
    sc = np.exp(a[:, [col["scale_0"], col["scale_1"], col["scale_2"]]])
    assert sc[7, 0] == pytest.approx(1.0, rel=0.2) and sc[7, 2] == pytest.approx(0.1 * sc[7, 0])  # flat disc
    for i in (0, 1, 7):
        w, x, y, z = a[i, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]]
        disc_axis = Rotation.from_quat([x, y, z, w]).apply([0, 0, 1.0])
        assert disc_axis == pytest.approx(N[i], abs=1e-5)
