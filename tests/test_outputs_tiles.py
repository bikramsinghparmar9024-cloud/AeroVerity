import json

import numpy as np
import pytest
import trimesh

from dronerecon.geo import LocalFrame, enu_to_ecef_matrix, lla_to_ecef
from dronerecon.outputs.mesh_io import SurfaceAttributes
from dronerecon.outputs.rasters import Grid
from dronerecon.outputs.tiles import raster_to_geojson, write_layers, write_tileset
from dronerecon.segmentation import Label

FRAME = LocalFrame(28.6, 77.2, 210.0)


def test_wgs84_helpers():
    assert lla_to_ecef(np.array([0.0]), np.array([0.0]), np.array([0.0]))[0] == pytest.approx([6378137.0, 0, 0])
    pole = lla_to_ecef(np.array([90.0]), np.array([0.0]), np.array([0.0]))[0]
    assert pole[2] == pytest.approx(6356752.314, abs=1e-3)
    M = enu_to_ecef_matrix(0.0, 0.0, 0.0)
    assert (M[:3, :3] @ [1, 0, 0]) == pytest.approx([0, 1, 0])  # east at (0,0) is +Y in ECEF
    assert (M[:3, :3] @ [0, 0, 1]) == pytest.approx([1, 0, 0])  # up is +X


def grid_mesh(n=41, step=5.0):
    xs = np.arange(n) * step - (n - 1) * step / 2
    X, Y = np.meshgrid(xs, xs)
    V = np.column_stack([X.ravel(), Y.ravel(), 3 * np.sin(X.ravel() / 20)])
    F = np.array([[j * n + i, j * n + i + 1, (j + 1) * n + i] for j in range(n - 1) for i in range(n - 1)]
                 + [[j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i] for j in range(n - 1) for i in range(n - 1)])
    return V, F


def test_tileset_structure_and_exact_placement(tmp_path):
    V, F = grid_mesh()  # 200 m x 200 m
    attrs = SurfaceAttributes(trust=np.zeros(len(V), np.uint8), labels=np.full(len(V), Label.ROAD, np.uint8))
    path = write_tileset(tmp_path / "tiles", V, F, None, attrs, FRAME, tile_m=100.0)
    ts = json.loads(path.read_text())
    assert ts["asset"]["version"] == "1.1"
    root = ts["root"]
    assert len(root["transform"]) == 16 and root["refine"] == "ADD"
    kids = root["children"]
    assert 4 <= len(kids) <= 9
    M = np.array(root["transform"]).reshape(4, 4).T  # column-major in the file
    total_faces = 0
    for kid in kids:
        g = trimesh.load(tmp_path / "tiles" / kid["content"]["uri"])
        mesh = list(g.geometry.values())[0] if hasattr(g, "geometry") else g
        total_faces += len(mesh.faces)
        # glTF (x, y, z) -> tileset z-up (x, -z, y) -> ECEF through the root transform
        v = np.asarray(mesh.vertices)
        enu = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])
        ecef = enu @ M[:3, :3].T + M[:3, 3]
        w, s, e, n, hmin, hmax = kid["boundingVolume"]["region"]
        assert np.all(np.isfinite(ecef))
        assert w < e and s < n and hmin <= hmax
    assert total_faces == len(F)

    # spot-check one vertex end to end
    lat, lon, h = FRAME.to_lla(V[:1])
    expected = lla_to_ecef(lat, lon, h)[0]
    g = trimesh.load(tmp_path / "tiles" / kids[0]["content"]["uri"])
    mesh = list(g.geometry.values())[0] if hasattr(g, "geometry") else g
    v = np.asarray(mesh.vertices)
    enu = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])
    ecef = enu @ M[:3, :3].T + M[:3, 3]
    assert np.min(np.linalg.norm(ecef - expected, axis=1)) < 0.01  # centimetre-exact placement


def test_geojson_layers_polygons_with_holes(tmp_path):
    grid = Grid(e0=0.0, n0=100.0, cell_m=1.0, width=100, height=100)
    classes = np.full((100, 100), Label.ROAD, np.uint8)
    classes[20:60, 30:70] = Label.BUILDING  # 40 x 40 m building
    classes[35:45, 45:55] = Label.VEGETATION  # courtyard: a hole in the building
    trust = np.zeros((100, 100), np.uint8)
    trust[:, 80:] = 255  # no data strip
    gj = raster_to_geojson(classes, grid, FRAME, {int(k): k.name.lower() for k in Label}, "class")
    by = {}
    for f in gj["features"]:
        by.setdefault(f["properties"]["class"], []).append(f)
    b = by["building"][0]
    assert len(b["geometry"]["coordinates"]) == 2  # outer ring + courtyard hole
    assert b["properties"]["area_m2"] == pytest.approx(40 * 40 - 10 * 10, rel=0.1)
    ring = b["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
    lat, lon, _ = FRAME.to_lla(np.array([[30.0, 80.0, 0.0]]))
    lons, lats = np.array(ring)[:, 0], np.array(ring)[:, 1]
    assert lons.min() == pytest.approx(lon[0], abs=2e-5) and lats.max() == pytest.approx(lat[0], abs=2e-5)
    files = write_layers(tmp_path, classes, trust, grid, FRAME)
    tr = json.loads(files["trust"].read_text())
    assert {f["properties"]["trust"] for f in tr["features"]} == {"Measured"}
