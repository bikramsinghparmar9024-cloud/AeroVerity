import json
import struct

import numpy as np
import pytest
import trimesh

from dronerecon.geo import LocalFrame
from dronerecon.outputs.mesh_io import TRUST_COLORS, SurfaceAttributes, write_glb, write_ply
from dronerecon.segmentation import Label
from dronerecon.types import DataError

FRAME = LocalFrame(28.6, 77.2, 215.0)


def pyramid():
    V = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [5, 5, 8]], float)
    F = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 2, 1], [0, 3, 2]])
    N = V - [5, 5, 3]
    attrs = SurfaceAttributes(trust=np.array([0, 0, 1, 2, 0], np.uint8),
                              sigma_m=np.array([0.1, 0.1, 0.6, np.nan, 0.2]),
                              labels=np.array([Label.ROAD, Label.ROAD, Label.BUILDING, 255, Label.BUILDING], np.uint8))
    return V, F, N, attrs


def read_glb(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack("<III", data[:12])
    jlen, jtype = struct.unpack("<II", data[12:20])
    gltf = json.loads(data[20:20 + jlen])
    blen, btype = struct.unpack("<II", data[20 + jlen:28 + jlen])
    return magic, version, length, len(data), gltf, data[28 + jlen:28 + jlen + blen], (jtype, btype)


def accessor(gltf, blob, idx, dtype, width):
    a = gltf["accessors"][idx]
    v = gltf["bufferViews"][a["bufferView"]]
    arr = np.frombuffer(blob[v["byteOffset"]:v["byteOffset"] + v["byteLength"]], dtype=dtype)
    return arr.reshape(a["count"], width) if width > 1 else arr


def test_glb_structure_attributes_and_axes(tmp_path):
    V, F, N, attrs = pyramid()
    p = write_glb(tmp_path / "model.glb", V, F, N, attrs, FRAME, title="Demo")
    magic, version, length, size, gltf, blob, types = read_glb(p)
    assert (magic, version, length) == (0x46546C67, 2, size)
    assert types == (0x4E4F534A, 0x004E4942)
    prim = gltf["meshes"][0]["primitives"][0]
    assert set(prim["attributes"]) == {"POSITION", "NORMAL", "COLOR_0", "_TRUST", "_SIGMA", "_CLASS"}
    pos = accessor(gltf, blob, prim["attributes"]["POSITION"], "<f4", 3)
    assert pos[4] == pytest.approx([5, 8, -5])  # (E, N, U) -> (x=E, y=U, z=-N)
    assert gltf["accessors"][prim["attributes"]["POSITION"]]["max"] == pytest.approx([10, 8, 0])
    trust = accessor(gltf, blob, prim["attributes"]["_TRUST"], "<f4", 1)
    assert trust.tolist() == [0, 0, 1, 2, 0]
    col = accessor(gltf, blob, prim["attributes"]["COLOR_0"], "u1", 4)
    assert (col[3, :3] == TRUST_COLORS[2]).all()
    assert gltf["asset"]["extras"]["origin_wgs84"]["lat"] == 28.6
    idx = accessor(gltf, blob, prim["indices"], "<u4", 1)
    assert idx.reshape(-1, 3).tolist() == F.tolist()


def test_glb_opens_in_a_standard_loader(tmp_path):
    V, F, N, attrs = pyramid()
    p = write_glb(tmp_path / "model.glb", V, F, N, attrs, FRAME, color_by="class")
    scene = trimesh.load(p)
    mesh = list(scene.geometry.values())[0] if hasattr(scene, "geometry") else scene
    assert len(mesh.vertices) == 5 and len(mesh.faces) == 6
    assert mesh.area == pytest.approx(trimesh.Trimesh(V, F).area, rel=1e-5)


def test_ply_mesh_and_points(tmp_path):
    V, F, N, attrs = pyramid()
    p = write_ply(tmp_path / "model.ply", V, F, N, attrs, FRAME)
    header = p.read_bytes().split(b"end_header")[0].decode()
    for token in ("property double x", "property uchar trust", "property float sigma_m", "property uchar class",
                  "element face 6", "origin_wgs84 lat=28.600000000"):
        assert token in header
    m = trimesh.load(p)
    assert len(m.vertices) == 5 and len(m.faces) == 6
    assert m.vertices[4] == pytest.approx(V[4])

    cloud = write_ply(tmp_path / "points.ply", V[:3], attrs=SurfaceAttributes(trust=attrs.trust[:3]))
    pc = trimesh.load(cloud)
    assert len(pc.vertices) == 3 and b"element face" not in cloud.read_bytes()


def test_bad_inputs_are_refused(tmp_path):
    V, F, N, attrs = pyramid()
    with pytest.raises(DataError, match="out of range"):
        write_glb(tmp_path / "x.glb", V, F + 10)
    with pytest.raises(DataError, match="trust has 2 values"):
        write_ply(tmp_path / "x.ply", V, attrs=SurfaceAttributes(trust=np.zeros(2, np.uint8)))
